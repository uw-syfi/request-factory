//! Immutable benchmark assets, resolved and verified before replay.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use sha2::{Digest, Sha256};

use crate::schema::AssetRef;

#[derive(Clone, Debug)]
pub struct LoadedAsset {
    pub path: PathBuf,
    pub bytes: Arc<[u8]>,
    pub media_type: String,
    pub sha256: String,
}

impl LoadedAsset {
    pub fn data_url(&self) -> String {
        let encoded = base64::engine::general_purpose::STANDARD.encode(self.bytes.as_ref());
        format!("data:{};base64,{encoded}", self.media_type)
    }
}

/// Per-run cache. The first read verifies the digest; subsequent requests
/// reuse the exact same immutable bytes without touching disk.
pub struct AssetStore {
    root: PathBuf,
    loaded: Mutex<HashMap<PathBuf, Arc<LoadedAsset>>>,
}

impl AssetStore {
    pub fn new(request_artifact: impl AsRef<Path>) -> Result<Self> {
        let artifact = request_artifact.as_ref();
        let root = artifact
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .canonicalize()
            .with_context(|| {
                format!(
                    "resolve request artifact directory for {}",
                    artifact.display()
                )
            })?;
        Ok(Self {
            root,
            loaded: Mutex::new(HashMap::new()),
        })
    }

    pub fn load(&self, reference: &AssetRef) -> Result<Arc<LoadedAsset>> {
        reference.validate("asset")?;
        let unresolved = Path::new(&reference.path);
        let path = if unresolved.is_absolute() {
            unresolved.to_path_buf()
        } else {
            self.root.join(unresolved)
        };
        let path = path
            .canonicalize()
            .with_context(|| format!("resolve asset {}", path.display()))?;
        if let Some(asset) = self
            .loaded
            .lock()
            .map_err(|_| anyhow!("asset cache lock poisoned"))?
            .get(&path)
            .cloned()
        {
            verify_reference(reference, &asset)?;
            return Ok(asset);
        }

        let bytes = std::fs::read(&path)
            .with_context(|| format!("read benchmark asset {}", path.display()))?;
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let media_type = reference
            .media_type
            .clone()
            .or_else(|| infer_media_type(&path).map(str::to_string))
            .ok_or_else(|| {
                anyhow!(
                    "cannot infer media type for {}; declare media_type in the asset reference",
                    path.display()
                )
            })?;
        let asset = Arc::new(LoadedAsset {
            path: path.clone(),
            bytes: Arc::from(bytes),
            media_type,
            sha256,
        });
        verify_reference(reference, &asset)?;
        self.loaded
            .lock()
            .map_err(|_| anyhow!("asset cache lock poisoned"))?
            .insert(path, asset.clone());
        Ok(asset)
    }

    pub fn preload<'a>(&self, references: impl IntoIterator<Item = &'a AssetRef>) -> Result<()> {
        for reference in references {
            self.load(reference)?;
        }
        Ok(())
    }
}

fn verify_reference(reference: &AssetRef, asset: &LoadedAsset) -> Result<()> {
    if reference
        .sha256
        .as_ref()
        .is_some_and(|expected| expected != &asset.sha256)
    {
        bail!(
            "sha256 mismatch for {}: expected {}, got {}",
            asset.path.display(),
            reference.sha256.as_deref().unwrap_or_default(),
            asset.sha256,
        );
    }
    if reference
        .media_type
        .as_ref()
        .is_some_and(|expected| expected != &asset.media_type)
    {
        bail!(
            "media type mismatch for {}: expected {}, got {}",
            asset.path.display(),
            reference.media_type.as_deref().unwrap_or_default(),
            asset.media_type,
        );
    }
    Ok(())
}

fn infer_media_type(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        "wav" => Some("audio/wav"),
        "mp3" => Some("audio/mpeg"),
        "flac" => Some("audio/flac"),
        "mp4" => Some("video/mp4"),
        "webm" => Some("video/webm"),
        "npy" => Some("application/x-npy"),
        "safetensors" => Some("application/x-safetensors"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "req_frontend_assets_{}_{}_{}",
            name,
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ))
    }

    #[test]
    fn assets_are_resolved_verified_cached_and_encoded() {
        let dir = fixture_dir("verified");
        std::fs::create_dir_all(&dir).unwrap();
        let artifact = dir.join("requests.jsonl");
        let image = dir.join("image.jpg");
        std::fs::write(&artifact, "").unwrap();
        std::fs::write(&image, b"jpeg bytes").unwrap();
        let digest = format!("{:x}", Sha256::digest(b"jpeg bytes"));
        let store = AssetStore::new(&artifact).unwrap();
        let reference = AssetRef {
            path: "image.jpg".into(),
            sha256: Some(digest),
            media_type: None,
        };

        let first = store.load(&reference).unwrap();
        let second = store.load(&reference).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.media_type, "image/jpeg");
        assert_eq!(first.data_url(), "data:image/jpeg;base64,anBlZyBieXRlcw==");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_changed_asset_is_rejected_before_replay() {
        let dir = fixture_dir("changed");
        std::fs::create_dir_all(&dir).unwrap();
        let artifact = dir.join("requests.jsonl");
        let image = dir.join("image.png");
        std::fs::write(&artifact, "").unwrap();
        std::fs::write(&image, b"changed").unwrap();
        let store = AssetStore::new(&artifact).unwrap();
        let reference = AssetRef {
            path: "image.png".into(),
            sha256: Some("0".repeat(64)),
            media_type: None,
        };

        let error = store.load(&reference).unwrap_err().to_string();
        assert!(error.contains("sha256 mismatch"));
        std::fs::remove_dir_all(dir).ok();
    }
}
