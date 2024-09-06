
import torch
from transformers import AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration, AutoProcessor, pipeline
from datasets import load_dataset

with_auto_pipeline = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

if with_auto_pipeline:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
else:
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

if with_auto_pipeline:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.feature_extractor,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(sample)
else:
    inputs = processor(sample["array"], return_tensors="pt")
    input_features = inputs.input_features

    generated_ids = model.generate(inputs=input_features)

    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]




print(result["text"])