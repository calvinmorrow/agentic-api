use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use super::output::CustomToolCall;
use crate::types::event::MessageStatus;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTextContent {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputImageContent {
    pub image_url: Option<String>,
    pub detail: Option<String>,
}

/// Content item inside a message input.
///
/// Uses an internally-tagged enum — serde consumes `"type"` for the variant
/// discriminant so the inner structs must NOT redeclare a `type_` field.
/// `output_text` and `reasoning_text` reuse `InputTextContent` since they
/// carry only a `text` field; they are preserved so vLLM sees the full history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    /// Assistant output text in rehydrated history.
    OutputText(InputTextContent),
    /// Reasoning step text in rehydrated history.
    ReasoningText(InputTextContent),
    /// Any other content type — drop silently.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputMessage {
    pub role: String,
    pub content: InputMessageContent,
}

impl Serialize for InputMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("InputMessage", 2)?;
        state.serialize_field("role", &self.role)?;
        match (self.role.as_str(), &self.content) {
            ("assistant", InputMessageContent::Text(text)) => {
                let content = vec![InputContent::OutputText(InputTextContent { text: text.clone() })];
                state.serialize_field("content", &content)?;
            }
            (_, content) => state.serialize_field("content", content)?,
        }
        state.end()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputMessageContent {
    Text(String),
    Parts(Vec<InputContent>),
}

/// Function-call result output — accepts a plain string, a content-parts array,
/// or a raw JSON object.
///
/// litellm (and the `OpenAI` Responses API spec) may emit `output` as either:
/// - a bare `String`, or
/// - a `List[ContentItem]` such as `[{"type": "input_text", "text": "..."}]`, or
/// - a raw JSON object like `{}` or `{"status": "ok"}`.
///
/// The untagged enum tries `Text` first (fast path for the majority of callers),
/// then `Parts` for arrays, then falls back to `Object` for plain JSON objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FunctionToolResultOutput {
    Text(String),
    Parts(Vec<InputContent>),
    /// Raw JSON object (e.g. `{}` or `{"status": "ok"}`) — litellm may emit
    /// non-string, non-array outputs for tool results.
    Object(Value),
}

impl FunctionToolResultOutput {
    /// Return the plain-text representation of this output.
    ///
    /// - `Text(s)` → `s`
    /// - `Object(v)` → JSON stringified `v`
    /// - `Parts` with a single `InputText` → its `text`
    /// - `Parts` with multiple items → concatenated `input_text` / `output_text` text values
    /// - `Parts` with no text items → empty string
    #[must_use]
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Object(v) => serde_json::to_string(v).unwrap_or_default(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    InputContent::InputText(c) | InputContent::OutputText(c) => Some(&c.text),
                    _ => None,
                })
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolResultMessage {
    pub call_id: String,
    pub output: FunctionToolResultOutput,
}

/// Client result for a freeform custom tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToolCallOutputMessage {
    pub call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub output: Value,
}

/// Wire wrapper for `function_call_output` input items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputItemFunctionCallOutput {
    #[serde(rename = "type")]
    #[serde(default = "default_fco_type")]
    pub type_: String,
    pub call_id: String,
    pub output: FunctionToolResultOutput,
}

fn default_fco_type() -> String {
    "function_call_output".into()
}

/// Wire wrapper for `function_call` input items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputItemFunctionCall {
    #[serde(rename = "type")]
    #[serde(default = "default_fc_type")]
    pub type_: String,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub call_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    #[serde(default)]
    pub arguments: String,
}

fn default_fc_type() -> String {
    "function_call".into()
}

/// Wire wrapper for `custom_tool_call` input items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputItemCustomToolCall {
    #[serde(rename = "type")]
    #[serde(default = "default_ctc_type")]
    pub type_: String,
    #[serde(default)]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
    #[serde(default)]
    pub call_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub input: String,
}

fn default_ctc_type() -> String {
    "custom_tool_call".into()
}

/// Wire wrapper for `custom_tool_call_output` input items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputItemCustomToolCallOutput {
    #[serde(rename = "type")]
    #[serde(default = "default_ctco_type")]
    pub type_: String,
    pub call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub output: Value,
}

fn default_ctco_type() -> String {
    "custom_tool_call_output".into()
}

/// Wire wrapper for `reasoning` input items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputItemReasoning {
    #[serde(rename = "type")]
    #[serde(default = "default_reasoning_type")]
    pub type_: String,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub content: Vec<super::output::ReasoningTextContent>,
    #[serde(default)]
    pub summary: Vec<serde_json::Value>,
    pub encrypted_content: Option<serde_json::Value>,
}

fn default_reasoning_type() -> String {
    "reasoning".into()
}

/// Wire wrapper for `message` input items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputItemMessage {
    #[serde(rename = "type")]
    #[serde(default = "default_message_type")]
    pub type_: String,
    #[serde(flatten)]
    pub message: InputMessage,
}

fn default_message_type() -> String {
    "message".into()
}

/// An input item for the Responses API `input` array.
///
/// Uses `untagged` so items **without** a `type` discriminator (e.g. bare
/// `{"role": "user", "content": "..."}`) deserialize as `Message`.  Tagged
/// items carry an explicit `type` field in their wire wrapper structs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    FunctionCallOutput(InputItemFunctionCallOutput),
    CustomToolCallOutput(InputItemCustomToolCallOutput),
    FunctionCall(InputItemFunctionCall),
    CustomToolCall(InputItemCustomToolCall),
    Reasoning(InputItemReasoning),
    Message(InputItemMessage),
}

impl From<FunctionToolResultMessage> for InputItem {
    fn from(msg: FunctionToolResultMessage) -> Self {
        InputItem::FunctionCallOutput(InputItemFunctionCallOutput {
            type_: default_fco_type(),
            call_id: msg.call_id,
            output: msg.output,
        })
    }
}

impl From<CustomToolCallOutputMessage> for InputItem {
    fn from(msg: CustomToolCallOutputMessage) -> Self {
        InputItem::CustomToolCallOutput(InputItemCustomToolCallOutput {
            type_: default_ctco_type(),
            call_id: msg.call_id,
            name: msg.name,
            output: msg.output,
        })
    }
}

impl From<super::output::FunctionToolCall> for InputItem {
    fn from(call: super::output::FunctionToolCall) -> Self {
        InputItem::FunctionCall(InputItemFunctionCall {
            type_: default_fc_type(),
            id: call.id,
            call_id: call.call_id,
            name: call.name,
            namespace: call.namespace,
            arguments: call.arguments,
        })
    }
}

impl From<CustomToolCall> for InputItem {
    fn from(call: CustomToolCall) -> Self {
        InputItem::CustomToolCall(InputItemCustomToolCall {
            type_: default_ctc_type(),
            id: call.id,
            status: call.status,
            call_id: call.call_id,
            name: call.name,
            input: call.input,
        })
    }
}

impl From<super::output::ReasoningOutput> for InputItem {
    fn from(r: super::output::ReasoningOutput) -> Self {
        InputItem::Reasoning(InputItemReasoning {
            type_: default_reasoning_type(),
            id: r.id,
            content: r.content,
            summary: r.summary,
            encrypted_content: r.encrypted_content,
        })
    }
}

impl From<InputMessage> for InputItem {
    fn from(msg: InputMessage) -> Self {
        InputItem::Message(InputItemMessage {
            type_: default_message_type(),
            message: msg,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_tool_result_output_deserialize_string() {
        let json = r#""simple text result""#;
        let output: FunctionToolResultOutput = serde_json::from_str(json).expect("deserialize string");
        match output {
            FunctionToolResultOutput::Text(s) => assert_eq!(s, "simple text result"),
            FunctionToolResultOutput::Parts(_) | FunctionToolResultOutput::Object(_) => panic!("expected Text variant"),
        }
    }

    #[test]
    fn function_tool_result_output_deserialize_parts_single_input_text() {
        let json = r#"[{"type": "input_text", "text": "tool result"}]"#;
        let output: FunctionToolResultOutput = serde_json::from_str(json).expect("deserialize parts");
        match output {
            FunctionToolResultOutput::Parts(parts) => {
                assert_eq!(parts.len(), 1);
                match &parts[0] {
                    InputContent::InputText(c) => assert_eq!(c.text, "tool result"),
                    other => panic!("expected InputText, got {other:?}"),
                }
            }
            FunctionToolResultOutput::Text(_) | FunctionToolResultOutput::Object(_) => panic!("expected Parts variant"),
        }
    }

    #[test]
    fn function_tool_result_output_deserialize_parts_multiple_items() {
        let json = r#"[{"type": "input_text", "text": "line 1"}, {"type": "input_text", "text": "line 2"}]"#;
        let output: FunctionToolResultOutput = serde_json::from_str(json).expect("deserialize parts");
        match output {
            FunctionToolResultOutput::Parts(parts) => {
                assert_eq!(parts.len(), 2);
            }
            FunctionToolResultOutput::Text(_) | FunctionToolResultOutput::Object(_) => panic!("expected Parts variant"),
        }
    }

    #[test]
    fn function_tool_result_output_deserialize_parts_unknown_type() {
        let json = r#"[{"type": "input_text", "text": "hello"}, {"type": "unknown_type", "foo": "bar"}]"#;
        let output: FunctionToolResultOutput = serde_json::from_str(json).expect("deserialize parts");
        match output {
            FunctionToolResultOutput::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                matches!(parts[0], InputContent::InputText(_));
                matches!(parts[1], InputContent::Unknown);
            }
            FunctionToolResultOutput::Text(_) | FunctionToolResultOutput::Object(_) => panic!("expected Parts variant"),
        }
    }

    #[test]
    fn function_tool_result_message_deserialize_string_output() {
        let json = r#"{"call_id": "call_abc", "output": "result text"}"#;
        let msg: FunctionToolResultMessage = serde_json::from_str(json).expect("deserialize message");
        assert_eq!(msg.call_id, "call_abc");
        assert_eq!(msg.output.to_text(), "result text");
    }

    #[test]
    fn function_tool_result_message_deserialize_list_output() {
        let json = r#"{"call_id": "call_123", "output": [{"type": "input_text", "text": "result text"}]}"#;
        let msg: FunctionToolResultMessage = serde_json::from_str(json).expect("deserialize message");
        assert_eq!(msg.call_id, "call_123");
        assert_eq!(msg.output.to_text(), "result text");
    }

    #[test]
    fn function_tool_result_message_deserialize_empty_list() {
        let json = r#"{"call_id": "call_empty", "output": []}"#;
        let msg: FunctionToolResultMessage = serde_json::from_str(json).expect("deserialize message");
        assert_eq!(msg.call_id, "call_empty");
        assert_eq!(msg.output.to_text(), "");
    }

    #[test]
    fn input_item_function_call_output_with_string_output() {
        let json = r#"{"type": "function_call_output", "call_id": "call_x", "output": "stdout"}"#;
        let item: InputItem = serde_json::from_str(json).expect("deserialize input item");
        match item {
            InputItem::FunctionCallOutput(msg) => {
                assert_eq!(msg.call_id, "call_x");
                assert_eq!(msg.output.to_text(), "stdout");
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn input_item_function_call_output_with_list_output() {
        let json = r#"{"type": "function_call_output", "call_id": "call_y", "output": [{"type": "input_text", "text": "stdout"}]}"#;
        let item: InputItem = serde_json::from_str(json).expect("deserialize input item");
        match item {
            InputItem::FunctionCallOutput(msg) => {
                assert_eq!(msg.call_id, "call_y");
                assert_eq!(msg.output.to_text(), "stdout");
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn input_item_function_call_output_json_round_trip() {
        let original = r#"{"type": "function_call_output", "call_id": "call_z", "output": [{"type": "input_text", "text": "hello world"}]}"#;
        let item: InputItem = serde_json::from_str(original).expect("deserialize");
        let serialized = serde_json::to_string(&item).expect("serialize");
        let deserialized: InputItem = serde_json::from_str(&serialized).expect("deserialize round-trip");
        match deserialized {
            InputItem::FunctionCallOutput(msg) => {
                assert_eq!(msg.call_id, "call_z");
                assert_eq!(msg.output.to_text(), "hello world");
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn responses_input_with_function_call_output_list() {
        let json = r#"[{"type": "message", "role": "user", "content": "hello"}, {"type": "function_call_output", "call_id": "call_1", "output": [{"type": "input_text", "text": "{\"status\":\"ok\"}"}]}]"#;
        let input: ResponsesInput = serde_json::from_str(json).expect("deserialize responses input");
        match input {
            ResponsesInput::Items(items) => {
                assert_eq!(items.len(), 2);
                match &items[1] {
                    InputItem::FunctionCallOutput(msg) => {
                        assert_eq!(msg.call_id, "call_1");
                        assert_eq!(msg.output.to_text(), "{\"status\":\"ok\"}");
                    }
                    other => panic!("expected FunctionCallOutput, got {other:?}"),
                }
            }
            ResponsesInput::Text(_) => panic!("expected Items variant"),
        }
    }

    #[test]
    fn to_text_with_multiple_parts_concatenates() {
        let json = r#"[{"type": "input_text", "text": "first"}, {"type": "input_text", "text": "second"}]"#;
        let output: FunctionToolResultOutput = serde_json::from_str(json).expect("deserialize");
        assert_eq!(output.to_text(), "first\nsecond");
    }

    #[test]
    fn to_text_skips_non_text_parts() {
        let json = r#"[{"type": "input_image", "image_url": "http://example.com/img.png"}, {"type": "input_text", "text": "after image"}]"#;
        let output: FunctionToolResultOutput = serde_json::from_str(json).expect("deserialize");
        assert_eq!(output.to_text(), "after image");
    }

    #[test]
    fn input_item_custom_tool_call_round_trip() {
        let json = r#"{"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "apply_patch", "input": "*** Begin Patch\n*** End Patch"}"#;
        let item: InputItem = serde_json::from_str(json).expect("deserialize custom tool call");
        match item {
            InputItem::CustomToolCall(call) => {
                assert_eq!(call.id, "ctc_1");
                assert_eq!(call.call_id, "call_1");
                assert_eq!(call.name, "apply_patch");
                assert_eq!(call.input, "*** Begin Patch\n*** End Patch");
            }
            other => panic!("expected CustomToolCall, got {other:?}"),
        }
    }

    #[test]
    fn input_item_custom_tool_call_output_round_trip() {
        let json = r#"{"type": "custom_tool_call_output", "call_id": "call_1", "name": "apply_patch", "output": {"status": "applied"}}"#;
        let item: InputItem = serde_json::from_str(json).expect("deserialize custom tool call output");
        match item {
            InputItem::CustomToolCallOutput(msg) => {
                assert_eq!(msg.call_id, "call_1");
                assert_eq!(msg.name, Some("apply_patch".to_string()));
            }
            other => panic!("expected CustomToolCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn from_custom_tool_call_to_input_item() {
        let call = CustomToolCall {
            id: "ctc_1".into(),
            status: Some(MessageStatus::Completed),
            call_id: "call_1".into(),
            name: "apply_patch".into(),
            input: "patch data".into(),
        };
        let item: InputItem = call.into();
        match item {
            InputItem::CustomToolCall(wrapper) => {
                assert_eq!(wrapper.id, "ctc_1");
                assert_eq!(wrapper.name, "apply_patch");
                assert_eq!(wrapper.input, "patch data");
            }
            other => panic!("expected CustomToolCall, got {other:?}"),
        }
    }
}
