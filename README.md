# CSE_368_Group12

1. Download Ollama at the following link: https://ollama.com/download

2. Pull model on command line with the following prompt: ollama pull llama3:8b

3. Download project zip from this GitHub.

4. Navigate to CSE_368_Group12-main\CSE_368_Group12-main\lightweight_crop_rotation_scheduler

5. Create virtual environment and download dependencies:
	- python (python3) -m venv crop-rotate
	- source crop-rotate/bin (Scripts)/activate
	- pip install pandas numpy ollama

6. Call crop scheduler python script:
	- python3 crop_rotation_scheduler.py -i INPUT_Field_Data_Tracker.csv -poly true -off 5
	- where -poly is optional and defaults to true
	- and -off can be any number
