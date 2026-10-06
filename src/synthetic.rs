//! Generated media inputs, for load that does not need a dataset.
//!
//! A benchmark replays recorded assets. A capacity run needs something else:
//! bytes of a chosen size, in a container the server will actually accept, at a
//! chosen rate. Random bytes will not do -- a server that decodes its inputs
//! rejects them, and then the run measures error handling instead of inference.
//!
//! So images, audio, and video are produced as real PNG, WAV, and MP4 files:
//! valid headers, valid checksums or indexes, and random pixels and samples.
//! Video uses independently decodable Motion JPEG frames, keeping the encoder
//! small and making every frame a seek point. See [`synthesize`].
//!
//! Content is a pure function of the seed and the shape, so the same spec yields
//! the same bytes and the store hands out one shared buffer. That is what makes
//! "non-unique content" cheap rather than a memory multiplier.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
use jpeg_encoder::{ColorType, Encoder};

use crate::assets::LoadedAsset;
use crate::schema::{Modality, SyntheticMedia};

/// Deterministic, cheap, and not cryptographic: this fills pixels, not keys.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let word = self.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for byte in bytes {
        a = (a + u32::from(*byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let mut checked = Vec::with_capacity(4 + body.len());
    checked.extend_from_slice(kind);
    checked.extend_from_slice(body);
    out.extend_from_slice(&checked);
    out.extend_from_slice(&crc32(&checked).to_be_bytes());
}

/// zlib stream of stored (uncompressed) deflate blocks.
///
/// Stored blocks mean no compressor, and random pixels would not compress
/// anyway. The point is a stream a decoder accepts, not a small one.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut offset = 0usize;
    while offset < raw.len() {
        let take = (raw.len() - offset).min(0xFFFF);
        let last = offset + take >= raw.len();
        out.push(u8::from(last));
        out.extend_from_slice(&(take as u16).to_le_bytes());
        out.extend_from_slice(&(!(take as u16)).to_le_bytes());
        out.extend_from_slice(&raw[offset..offset + take]);
        offset += take;
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

/// A real 8-bit RGB PNG with random pixels.
fn png(width: u32, height: u32, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64(seed);
    let stride = width as usize * 3;
    // One filter byte per scanline, filter type 0 (None).
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    let mut row = vec![0u8; stride];
    for _ in 0..height {
        rng.fill(&mut row);
        raw.push(0);
        raw.extend_from_slice(&row);
    }
    let mut out = Vec::with_capacity(raw.len() + 64);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, truecolour, no interlace
    png_chunk(&mut out, b"IHDR", &ihdr);
    png_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    png_chunk(&mut out, b"IEND", &[]);
    out
}

/// A real mono 16-bit PCM WAV with random samples.
fn wav(sample_rate_hz: u32, duration_ms: u32, seed: u64) -> Vec<u8> {
    let samples = (u64::from(sample_rate_hz) * u64::from(duration_ms) / 1000) as usize;
    let data_len = samples * 2;
    let mut data = vec![0u8; data_len];
    SplitMix64(seed).fill(&mut data);
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate_hz.to_le_bytes());
    out.extend_from_slice(&(sample_rate_hz * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

fn mp4_box(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    debug_assert!(body.len() <= u32::MAX as usize - 8);
    out.extend_from_slice(&((body.len() + 8) as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
}

fn full_mp4_box(out: &mut Vec<u8>, kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) {
    let mut full_body = Vec::with_capacity(body.len() + 4);
    full_body.push(version);
    full_body.extend_from_slice(&flags.to_be_bytes()[1..]);
    full_body.extend_from_slice(body);
    mp4_box(out, kind, &full_body);
}

fn mp4_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn mp4_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn mp4_identity_matrix(out: &mut Vec<u8>) {
    for value in [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
        mp4_u32(out, value);
    }
}

fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 8);
    mp4_box(&mut out, kind, body);
    out
}

fn full_boxed(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 12);
    full_mp4_box(&mut out, kind, version, flags, body);
    out
}

/// A decodable MP4 containing independently decodable Motion JPEG frames.
///
/// Each JPEG is zero-padded to a shape-derived slot. JPEG decoders ignore data
/// after the end marker, so the MP4 remains valid while its byte size stays an
/// exact function of shape rather than of random pixels' compression ratio.
fn mp4(width: u32, height: u32, frames: u32, fps: f64, seed: u64) -> Result<Vec<u8>> {
    const TIMESCALE: u32 = 90_000;
    let width_u16 =
        u16::try_from(width).map_err(|_| anyhow!("synthetic video width exceeds MP4 limits"))?;
    let height_u16 =
        u16::try_from(height).map_err(|_| anyhow!("synthetic video height exceeds MP4 limits"))?;
    let pixel_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| anyhow!("synthetic video frame size overflows this platform"))?;
    // Quality-75 JPEGs are substantially smaller than RGB. Two RGB-sized
    // buffers plus fixed headroom provide a conservative, predictable slot.
    let frame_slot = pixel_bytes
        .checked_mul(2)
        .and_then(|value| value.checked_add(2_048))
        .ok_or_else(|| anyhow!("synthetic video frame slot overflows this platform"))?;
    let frame_slot_u32 = u32::try_from(frame_slot)
        .map_err(|_| anyhow!("synthetic video frame exceeds MP4 sample limits"))?;
    let media_bytes = (frames as usize)
        .checked_mul(frame_slot)
        .ok_or_else(|| anyhow!("synthetic video payload size overflows this platform"))?;
    if media_bytes > u32::MAX as usize - 8 {
        bail!("synthetic video exceeds the MP4 mdat box limit");
    }
    let total_bytes = media_bytes
        .checked_add(603)
        .ok_or_else(|| anyhow!("synthetic video size overflows this platform"))?;
    let sample_delta = (f64::from(TIMESCALE) / fps)
        .round()
        .clamp(1.0, f64::from(u32::MAX)) as u32;
    let duration = frames
        .checked_mul(sample_delta)
        .ok_or_else(|| anyhow!("synthetic video duration exceeds MP4 limits"))?;

    let mut pixels = vec![0u8; pixel_bytes];
    let mut media = Vec::new();
    media.try_reserve_exact(media_bytes)?;
    let mut rng = SplitMix64(seed);
    for _ in 0..frames {
        rng.fill(&mut pixels);
        let mut jpeg = Vec::new();
        Encoder::new(&mut jpeg, 75)
            .encode(&pixels, width_u16, height_u16, ColorType::Rgb)
            .map_err(|error| anyhow!("failed to encode synthetic video frame: {error}"))?;
        if jpeg.len() > frame_slot {
            bail!(
                "synthetic JPEG frame is {} bytes, exceeding its {frame_slot}-byte MP4 slot",
                jpeg.len()
            );
        }
        media.extend_from_slice(&jpeg);
        media.resize(media.len() + frame_slot - jpeg.len(), 0);
    }

    let ftyp = boxed(b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41");
    let chunk_offset = u32::try_from(ftyp.len() + 8).unwrap();
    let mdat = boxed(b"mdat", &media);

    let mut mvhd_body = Vec::with_capacity(96);
    for value in [0, 0, TIMESCALE, duration, 0x0001_0000] {
        mp4_u32(&mut mvhd_body, value);
    }
    mp4_u16(&mut mvhd_body, 0x0100);
    mp4_u16(&mut mvhd_body, 0);
    mp4_u32(&mut mvhd_body, 0);
    mp4_u32(&mut mvhd_body, 0);
    mp4_identity_matrix(&mut mvhd_body);
    for _ in 0..6 {
        mp4_u32(&mut mvhd_body, 0);
    }
    mp4_u32(&mut mvhd_body, 2);
    let mvhd = full_boxed(b"mvhd", 0, 0, &mvhd_body);

    let mut tkhd_body = Vec::with_capacity(80);
    for value in [0, 0, 1, 0, duration, 0, 0] {
        mp4_u32(&mut tkhd_body, value);
    }
    for _ in 0..4 {
        mp4_u16(&mut tkhd_body, 0);
    }
    mp4_identity_matrix(&mut tkhd_body);
    mp4_u32(&mut tkhd_body, width << 16);
    mp4_u32(&mut tkhd_body, height << 16);
    let tkhd = full_boxed(b"tkhd", 0, 3, &tkhd_body);

    let mut mdhd_body = Vec::with_capacity(20);
    for value in [0, 0, TIMESCALE, duration] {
        mp4_u32(&mut mdhd_body, value);
    }
    mp4_u16(&mut mdhd_body, 0x55c4); // ISO-639-2/T `und`
    mp4_u16(&mut mdhd_body, 0);
    let mdhd = full_boxed(b"mdhd", 0, 0, &mdhd_body);

    let mut hdlr_body = Vec::with_capacity(33);
    mp4_u32(&mut hdlr_body, 0);
    hdlr_body.extend_from_slice(b"vide");
    for _ in 0..3 {
        mp4_u32(&mut hdlr_body, 0);
    }
    hdlr_body.extend_from_slice(b"VideoHandler\0");
    let hdlr = full_boxed(b"hdlr", 0, 0, &hdlr_body);

    let mut vmhd_body = Vec::with_capacity(8);
    for _ in 0..4 {
        mp4_u16(&mut vmhd_body, 0);
    }
    let vmhd = full_boxed(b"vmhd", 0, 1, &vmhd_body);
    let url = full_boxed(b"url ", 0, 1, &[]);
    let mut dref_body = Vec::with_capacity(16);
    mp4_u32(&mut dref_body, 1);
    dref_body.extend_from_slice(&url);
    let dinf = boxed(b"dinf", &full_boxed(b"dref", 0, 0, &dref_body));

    let mut entry_body = Vec::with_capacity(78);
    entry_body.extend_from_slice(&[0; 6]);
    mp4_u16(&mut entry_body, 1);
    mp4_u16(&mut entry_body, 0);
    mp4_u16(&mut entry_body, 0);
    for _ in 0..3 {
        mp4_u32(&mut entry_body, 0);
    }
    mp4_u16(&mut entry_body, width_u16);
    mp4_u16(&mut entry_body, height_u16);
    mp4_u32(&mut entry_body, 0x0048_0000);
    mp4_u32(&mut entry_body, 0x0048_0000);
    mp4_u32(&mut entry_body, 0);
    mp4_u16(&mut entry_body, 1);
    let name = b"req-frontend mjpeg";
    entry_body.push(name.len() as u8);
    entry_body.extend_from_slice(name);
    entry_body.resize(entry_body.len() + 31 - name.len(), 0);
    mp4_u16(&mut entry_body, 0x0018);
    mp4_u16(&mut entry_body, u16::MAX);
    debug_assert_eq!(entry_body.len(), 78);
    let entry = boxed(b"mjpg", &entry_body);

    let mut stsd_body = Vec::with_capacity(90);
    mp4_u32(&mut stsd_body, 1);
    stsd_body.extend_from_slice(&entry);
    let stsd = full_boxed(b"stsd", 0, 0, &stsd_body);
    let mut stts_body = Vec::with_capacity(12);
    for value in [1, frames, sample_delta] {
        mp4_u32(&mut stts_body, value);
    }
    let stts = full_boxed(b"stts", 0, 0, &stts_body);
    let mut stsc_body = Vec::with_capacity(16);
    for value in [1, 1, frames, 1] {
        mp4_u32(&mut stsc_body, value);
    }
    let stsc = full_boxed(b"stsc", 0, 0, &stsc_body);
    let mut stsz_body = Vec::with_capacity(8);
    mp4_u32(&mut stsz_body, frame_slot_u32);
    mp4_u32(&mut stsz_body, frames);
    let stsz = full_boxed(b"stsz", 0, 0, &stsz_body);
    let mut stco_body = Vec::with_capacity(8);
    mp4_u32(&mut stco_body, 1);
    mp4_u32(&mut stco_body, chunk_offset);
    let stco = full_boxed(b"stco", 0, 0, &stco_body);

    let stbl = boxed(b"stbl", &[stsd, stts, stsc, stsz, stco].concat());
    let minf = boxed(b"minf", &[vmhd, dinf, stbl].concat());
    let mdia = boxed(b"mdia", &[mdhd, hdlr, minf].concat());
    let trak = boxed(b"trak", &[tkhd, mdia].concat());
    let moov = boxed(b"moov", &[mvhd, trak].concat());

    let mut out = Vec::new();
    out.try_reserve_exact(total_bytes)?;
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&mdat);
    out.extend_from_slice(&moov);
    debug_assert_eq!(out.len(), total_bytes);
    Ok(out)
}

/// Build one synthetic media input.
///
/// `unique_salt` is mixed in only when the spec pins no seed, so a trace that
/// wants byte-identical content across requests gets exactly that, and one that
/// wants distinct content per request gets that instead.
pub(crate) fn synthesize(
    spec: &SyntheticMedia,
    modality: Modality,
    unique_salt: &str,
) -> Result<(Vec<u8>, &'static str)> {
    let seed = spec.seed.unwrap_or_else(|| {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in unique_salt.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        hash
    });
    let need = |value: Option<u32>, what: &str| -> Result<u32> {
        value.ok_or_else(|| anyhow!("synthetic {modality:?} input requires {what}"))
    };
    match modality {
        Modality::Image => Ok((
            png(
                need(spec.width, "width")?,
                need(spec.height, "height")?,
                seed,
            ),
            "image/png",
        )),
        Modality::Audio => Ok((
            wav(
                need(spec.sample_rate_hz, "sample_rate_hz")?,
                need(spec.duration_ms, "duration_ms")?,
                seed,
            ),
            "audio/wav",
        )),
        Modality::Video => Ok((
            mp4(
                need(spec.width, "width")?,
                need(spec.height, "height")?,
                need(spec.frames, "frames")?,
                spec.fps.unwrap_or(1.0),
                seed,
            )?,
            "video/mp4",
        )),
        other => bail!("synthetic content is not defined for {other:?}"),
    }
}

/// Per-run cache of generated media, mirroring [`crate::assets::AssetStore`].
///
/// Keyed by the shape and seed, so a trace whose requests share one spec
/// generates once and reuses one `Arc` -- the difference between "non-unique
/// content is cheap" and "every request costs another megabyte".
#[derive(Default)]
pub(crate) struct SyntheticStore {
    generated: Mutex<HashMap<String, Arc<LoadedAsset>>>,
}

impl SyntheticStore {
    pub(crate) fn build(
        &self,
        spec: &SyntheticMedia,
        modality: Modality,
        unique_salt: &str,
    ) -> Result<Arc<LoadedAsset>> {
        let key = format!(
            "{modality:?}|{:?}|{:?}x{:?}|{:?}@{:?}|{:?}fps{:?}|{}",
            spec.seed,
            spec.width,
            spec.height,
            spec.duration_ms,
            spec.sample_rate_hz,
            spec.frames,
            spec.fps,
            // Only part of the key when the spec pins no seed; otherwise every
            // request with the same shape shares one buffer.
            if spec.seed.is_some() { "" } else { unique_salt },
        );
        if let Some(found) = self
            .generated
            .lock()
            .map_err(|_| anyhow!("synthetic cache lock poisoned"))?
            .get(&key)
            .cloned()
        {
            return Ok(found);
        }
        let (bytes, media_type) = synthesize(spec, modality, unique_salt)?;
        let asset = Arc::new(LoadedAsset {
            path: format!("synthetic:{key}").into(),
            bytes: Arc::from(bytes.into_boxed_slice()),
            media_type: media_type.to_string(),
            // Generated content has no recorded digest to verify against; the
            // shape and seed in the key are its provenance.
            sha256: String::new(),
        });
        self.generated
            .lock()
            .map_err(|_| anyhow!("synthetic cache lock poisoned"))?
            .insert(key, asset.clone());
        Ok(asset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_spec(seed: Option<u64>) -> SyntheticMedia {
        SyntheticMedia {
            seed,
            width: Some(4),
            height: Some(3),
            ..blank()
        }
    }

    fn blank() -> SyntheticMedia {
        SyntheticMedia {
            seed: None,
            width: None,
            height: None,
            sample_rate_hz: None,
            duration_ms: None,
            frames: None,
            fps: None,
        }
    }

    #[test]
    fn png_is_a_real_png_with_correct_chunk_checksums() {
        let bytes = png(4, 3, 7);
        assert_eq!(
            &bytes[..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );
        // Walk the chunk list, verifying every CRC, and confirm IHDR/IEND frame it.
        let mut offset = 8usize;
        let mut kinds = Vec::new();
        while offset + 8 <= bytes.len() {
            let len = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            let kind = &bytes[offset + 4..offset + 8];
            let body_end = offset + 8 + len;
            let stored = u32::from_be_bytes(bytes[body_end..body_end + 4].try_into().unwrap());
            assert_eq!(crc32(&bytes[offset + 4..body_end]), stored, "chunk crc");
            kinds.push(String::from_utf8_lossy(kind).to_string());
            offset = body_end + 4;
        }
        assert_eq!(kinds, vec!["IHDR", "IDAT", "IEND"]);
        assert_eq!(offset, bytes.len(), "no trailing bytes");
    }

    #[test]
    fn wav_header_describes_the_samples_that_follow() {
        let bytes = wav(8_000, 250, 1);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let data_len = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
        // 8 kHz * 0.25 s * 2 bytes = 4000
        assert_eq!(data_len, 4_000);
        assert_eq!(bytes.len(), 44 + data_len);
        let riff_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        assert_eq!(riff_len, bytes.len() - 8);
    }

    #[test]
    fn mp4_boxes_and_sample_table_describe_every_frame() {
        let bytes = mp4(64, 48, 4, 2.0, 3).unwrap();
        let frame_slot = 64 * 48 * 3 * 2 + 2_048;
        assert_eq!(bytes.len(), 603 + 4 * frame_slot);

        let mut offset = 0usize;
        let mut kinds = Vec::new();
        while offset < bytes.len() {
            let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            kinds.push(String::from_utf8_lossy(&bytes[offset + 4..offset + 8]).to_string());
            assert!(size >= 8 && offset + size <= bytes.len());
            offset += size;
        }
        assert_eq!(kinds, ["ftyp", "mdat", "moov"]);
        assert_eq!(offset, bytes.len());
        assert_eq!(&bytes[36..38], &[0xff, 0xd8], "first sample is JPEG");

        let stsz = bytes
            .windows(4)
            .position(|window| window == b"stsz")
            .expect("sample-size box");
        assert_eq!(
            u32::from_be_bytes(bytes[stsz + 8..stsz + 12].try_into().unwrap()),
            frame_slot as u32
        );
        assert_eq!(
            u32::from_be_bytes(bytes[stsz + 12..stsz + 16].try_into().unwrap()),
            4
        );
    }

    #[test]
    fn mp4_is_decodable_when_ffmpeg_is_available() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = match Command::new("ffmpeg")
            .args(["-v", "error", "-i", "pipe:0", "-f", "null", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            Err(error) => panic!("failed to start ffmpeg: {error}"),
        };
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&mp4(32, 24, 3, 2.0, 5).unwrap())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "ffmpeg rejected MP4: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn a_pinned_seed_makes_content_reproducible_and_size_follows_the_shape() {
        let (small, _) = synthesize(&image_spec(Some(9)), Modality::Image, "a").unwrap();
        let (again, _) = synthesize(&image_spec(Some(9)), Modality::Image, "b").unwrap();
        // Same seed, different request: byte-identical, which is what a
        // non-unique-content run asks for.
        assert_eq!(small, again);

        let bigger = SyntheticMedia {
            width: Some(64),
            height: Some(64),
            ..image_spec(Some(9))
        };
        let (large, _) = synthesize(&bigger, Modality::Image, "a").unwrap();
        assert!(
            large.len() > small.len() * 10,
            "size tracks the requested shape"
        );
    }

    #[test]
    fn without_a_seed_each_request_gets_its_own_bytes() {
        let (first, _) = synthesize(&image_spec(None), Modality::Image, "req-1").unwrap();
        let (second, _) = synthesize(&image_spec(None), Modality::Image, "req-2").unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn the_store_reuses_one_buffer_for_a_pinned_seed() {
        let store = SyntheticStore::default();
        let first = store
            .build(&image_spec(Some(4)), Modality::Image, "req-1")
            .unwrap();
        let second = store
            .build(&image_spec(Some(4)), Modality::Image, "req-2")
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second), "shared, not regenerated");

        let unseeded = SyntheticStore::default();
        let a = unseeded
            .build(&image_spec(None), Modality::Image, "req-1")
            .unwrap();
        let b = unseeded
            .build(&image_spec(None), Modality::Image, "req-2")
            .unwrap();
        assert!(!Arc::ptr_eq(&a, &b), "distinct content stays distinct");
    }

    #[test]
    fn a_data_url_carries_the_generated_media_type() {
        let store = SyntheticStore::default();
        let audio = SyntheticMedia {
            sample_rate_hz: Some(16_000),
            duration_ms: Some(100),
            ..blank()
        };
        let built = store.build(&audio, Modality::Audio, "r").unwrap();
        assert!(built.data_url().starts_with("data:audio/wav;base64,"));
    }
}
