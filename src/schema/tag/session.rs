//! Per-row values added by the `session` tag to an independent-family format.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestSession {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub prefix_kv: Option<u32>,
    #[serde(default)]
    pub tool_wait_after_ms: Option<f64>,
}

impl RequestSession {
    pub fn validate(&self, at: &str) -> Result<()> {
        let prefix_kv = self.prefix_kv.unwrap_or(0);
        let tool_wait_after_ms = self.tool_wait_after_ms.unwrap_or(0.0);
        if self.session_id.as_deref().is_none_or(str::is_empty)
            && (prefix_kv != 0 || tool_wait_after_ms != 0.0)
        {
            bail!("{at}: prefix_kv/tool_wait_after_ms requires a non-empty session_id");
        }
        if !tool_wait_after_ms.is_finite() || tool_wait_after_ms < 0.0 {
            bail!("{at}: tool_wait_after_ms must be finite and non-negative");
        }
        Ok(())
    }
}
