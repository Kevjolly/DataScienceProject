import requests
import json
  
URL_BASE = "https://82d4gowgcb.execute-api.eu-west-2.amazonaws.com/astronomy-ads"
URL_BASEDATA = "/data"
URL_NEWDATA = "/handledata"
URL_GETBASEDATA_PARAMS = ["dataset", "name", "nbr"]
URL_GETNEWDATA_PARAMS = ["starClass", "number", "phased"]

DATASETS = ["mallat", "starlightcurves"]
FILENAMES = [["MALLAT_TEST", "MALLAT_TRAIN"],["test.csv", "train.csv"]]
  
#def getBaseData(dataset, name, nbrData=None, classAllowed=None, classDisallowed=None, randomData=False):
def getBaseData(dataset, name):
	""" dataset -> string name of the dataset
		name -> the name of the set in the dataset 
		nbrData -> int, maximum number of data rows you want (might be smaller if there isn't enough data rows)
		classAllowed -> a list of the classes (int) you want
		classDisallowed -> a list of the classes (int) you don't want
		randomData -> boolean informing that you want random data rows among the ones you chose to want """
	numberGet = 0
	remainingLines = 1
	results = []
	nbrResults = 0
	while (remainingLines != 0):
		PARAMS = {URL_GETBASEDATA_PARAMS[0]:dataset, URL_GETBASEDATA_PARAMS[1]:name,  URL_GETBASEDATA_PARAMS[2]:numberGet}
		r = requests.get(url = URL_BASE+URL_BASEDATA, params = PARAMS)  
		data = r.json()
		print("Retrieving data from ", data['key'], " Part ", str(numberGet))
		remainingLines = data["remainingLines"]
		results += data["results"]
		nbrResults += data["nbrResults"]
		numberGet += 1
	print("Finished retrieving data: ", nbrResults, " results.")
	return results

def getDatasetsAndFilenames():
	return (DATASETS, FILENAMES)

def postNewData(phased, label, times, magnitude, magnitude_error):
	# data to be sent to api
	if not isinstance(phased, bool):
		return (-1, "phase needs to be a boolean")
	if not isinstance(label, str):
		return (-1, "label needs to be a string")
	if not isinstance(times, list):
		return (-1, "times needs to be a list")
	if not isinstance(magnitude, list):
		return (-1, "magnitude needs to be a list")
	if not isinstance(magnitude_error, list):
		return (-1, "magnitude_error needs to be a list")
	data = {"phased":phased, "label":label, "times":times, "mags":magnitude, "mags_err":magnitude_error} 
	#data = {"label":"star3", "times": [0, 2, 3], "mags": [0.234, 0.1, 0.1], "mags_err": [0.4536, 0.43, 0.222]}

	# sending post request and saving response as response object 
	r = requests.post(url = URL_BASE+URL_NEWDATA, json=data) 
	return (r.status_code, r.json()['body'])

def getNewData(phased, number, starClass=None):
	if not isinstance(phased, bool):
		return "phased needs to be a boolean"
	if not isinstance(number, int):
		return "number needs to be an int"
	if starClass is None:
		starClass = "noClass"
	PARAMS = {URL_GETNEWDATA_PARAMS[0]:starClass, URL_GETNEWDATA_PARAMS[1]:number,  URL_GETNEWDATA_PARAMS[2]:phased}
	r = requests.get(url = URL_BASE+URL_NEWDATA, params = PARAMS)  
	data = r.json()
	print("Request returned with status_code: ", data["statusCode"])
	print("Finished retrieving data: ", data["numberResults"])
	return data["results"]



