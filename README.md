# Set Up Guide
1. Set up a project in google cloud using GCP console
	1. Create a new project
	2. Enable Google NL API
	3. Create a service account
	4. Download a private key as JSON
	
    More infomation about GCP project https://cloud.google.com/resource-manager/docs/creating-managing-projects

2. Open Terminal and set the environment variable GOOGLE_APPLICATION_CREDENTIALS to downloaded JSON file's path.

   For Linux or Mac`export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"`
   
   For Windows `$env:GOOGLE_APPLICATION_CREDENTIALS="[PATH]"`
   
   Change "[PATH]" with JSON file's location

3. Install Pre-req libraries
 'pip install google-cloud-language'
  sometimes more based on other installed libraries.

4. Run the `sample.py` in python environment opened from the terminal of step 2 	

### Sentiment analysis of a txt file

Use bellow mentioned code to get the sentiment score of a txt file

`python sentiment-analysis.py positive-sample.txt`
