import argparse
import os
from helpers import *
from faster_whisper import WhisperModel
import whisperx
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
import uuid
unique_id = str(uuid.uuid4())
mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

# New arguments to specify output file paths
parser.add_argument(
    "--txt-output",
    dest="txt_output",
    default=None,
    help="Path to save the .txt output file",
)

parser.add_argument(
    "--srt-output",
    dest="srt_output",
    default=None,
    help="Path to save the .srt output file",
)


args = parser.parse_args()

if args.stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio


# Transcribe the audio file
if args.batch_size != 0:
    from transcription_helpers import transcribe_batched

    whisper_results, language = transcribe_batched(
        vocal_target,
        args.language,
        args.batch_size,
        args.model_name,
        mtypes[args.device],
        args.suppress_numerals,
        args.device,
    )
else:
    from transcription_helpers import transcribe

    whisper_results, language = transcribe(
        vocal_target,
        args.language,
        args.model_name,
        mtypes[args.device],
        args.suppress_numerals,
        args.device,
    )

if language in wav2vec2_langs:
    alignment_model, metadata = whisperx.load_align_model(
        language_code=language, device=args.device
    )
    result_aligned = whisperx.align(
        whisper_results, alignment_model, metadata, vocal_target, args.device
    )
    word_timestamps = filter_missing_timestamps(
        result_aligned["word_segments"],
        initial_timestamp=whisper_results[0].get("start"),
        final_timestamp=whisper_results[-1].get("end"),
    )
    # clear gpu vram
    del alignment_model
    torch.cuda.empty_cache()
else:
    assert (
        args.batch_size == 0  # TODO: add a better check for word timestamps existence
    ), (
        f"Unsupported language: {language}, use --batch_size to 0"
        " to generate word timestamps using whisper directly and fix this error."
    )
    word_timestamps = []
    for segment in whisper_results:
        for word in segment["words"]:
            word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})
# Extracting the 'text' values
# Modified code to store start and end timestamps along with the text
transcription_results = []

for segment in whisper_results:
    text = segment['text']
    start_time = segment['start']
    end_time = segment['end']
    transcription_results.append({'text': text, 'start': start_time, 'end': end_time})

# Extracting the 'text', 'start', and 'end' values
text_start_end_values = [(result['text'], result['start'], result['end']) for result in transcription_results]
print(text_start_end_values)

# convert audio to mono for NeMo combatibility
sound = AudioSegment.from_file(vocal_target).set_channels(1)
ROOT = os.getcwd()
temp_path = os.path.join("../../workspace/workspace", "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping


speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logging.warning(
        f"Punctuation restoration is not available for {language} language. Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)


txt_file_path = args.txt_output 
#with open(txt_file_path, "w", encoding="utf-8-sig") as f:
#    get_speaker_aware_transcript(ssm, f)

# Write the subtitles to a .srt file
srt_file_path = args.srt_output 
with open(srt_file_path, "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)
    
print(txt_file_path)

import json

# Path to save the JSON file
json_file_path = os.path.splitext(srt_file_path)[0] + ".json"
# Create a list to store the transcription results with start, end, and text
transcription_data = []

# Populate the list with transcription results
for result in whisper_results:
    transcription_data.append({
        "start": result["start"],
        "end": result["end"],
        "text": result["text"]
    })

# Write the transcription data to the JSON file
with open(json_file_path, "w", encoding="utf-8") as json_file:
    json.dump(transcription_data, json_file, indent=4)

print(f"JSON file saved at: {json_file_path}")