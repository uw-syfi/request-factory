//! Generated media inputs, for load that does not need a dataset.
//!
//! A benchmark replays recorded assets. A capacity run needs something else:
//! bytes of a chosen size, in a container the server will actually accept, at a
//! chosen rate. Random bytes will not do -- a server that decodes its inputs
//! rejects them, and then the run measures error handling instead of inference.
//!
//! So images and audio are produced as real PNG and WAV files: valid headers,
//! valid checksums, random pixels and samples. Video is emitted as a
//! structurally well-formed MP4 box tree whose sample payload is random; that is
//! enough for a server that forwards or measures the upload, and deliberately
//! not enough for one that decodes frames. See [`synthesize`].
//!
//! Content is a pure function of the seed and the shape, so the same spec yields
//! the same bytes and the store hands out one shared buffer. That is what makes
//! "non-unique content" cheap rather than a memory multiplier.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};

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
    out.extend_from_slice(&((body.len() + 8) as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
}

/// A well-formed MP4 box tree with a random `mdat` payload.
///
/// Honest about what this is: the container parses and its size scales with the
/// requested shape, which is what an upload-path or throughput measurement
/// needs. It carries no `moov` track description, so a server that actually
/// decodes frames will reject it -- use a recorded asset for that.
fn mp4(width: u32, height: u32, frames: u32, seed: u64) -> Vec<u8> {
    let payload_len = (width as usize * height as usize * 3 / 8).max(1) * frames as usize;
    let mut payload = vec![0u8; payload_len];
    SplitMix64(seed).fill(&mut payload);
    let mut out = Vec::with_capacity(payload_len + 64);
    mp4_box(&mut out, b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41");
    mp4_box(&mut out, b"mdat", &payload);
    out
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
                seed,
            ),
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
    fn mp4_boxes_declare_their_own_lengths() {
        let bytes = mp4(64, 64, 4, 3);
        let mut offset = 0usize;
        let mut kinds = Vec::new();
        while offset + 8 <= bytes.len() {
            let len = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            assert!(
                len >= 8 && offset + len <= bytes.len(),
                "box length in range"
            );
            kinds.push(String::from_utf8_lossy(&bytes[offset + 4..offset + 8]).to_string());
            offset += len;
        }
        assert_eq!(kinds, vec!["ftyp", "mdat"]);
        assert_eq!(offset, bytes.len());
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
