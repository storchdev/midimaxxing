Streamline converting piano audio to midi using bytedance's [docker container for piano-transcription](https://replicate.com/bytedance/piano-transcription?input=docker).

## usage
- Have docker and run `rundocker.sh`
- Install python dependencies (you only need `requests`)
- Run `main.py myaudio.mp3`
- Wait for the program the finish and `transcription.mid` to show up in your cwd

## notes
- Python code is mostly written by Gemini 3 Pro

