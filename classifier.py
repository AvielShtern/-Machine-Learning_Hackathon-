from Preproccesor import *

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
pickle_in = open("TrainedSerializedModel.bin", "rb")
MODELS = pickle.load(pickle_in)
pickle_in.close()


def predict(cvs_path):
    X = ModelProcessor(cvs_path).total_df.values
    return MODELS.classifier.predict(X)


def send_police_cars(date):
    return MODELS.police_cars.predict(date)

