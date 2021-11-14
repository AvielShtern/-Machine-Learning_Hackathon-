from Preproccesor import *

if __name__ == "__main__":
    pre = ModelProcessor(DATA_PATH, True)
    classifier = pre.evaluation()
    pickle_out = open("TrainedSerializedModel.bin", "wb")
    trained_models = Serialization(classifier, SendPoliceCars(train_model_send_police()))
    pickle.dump(trained_models, pickle_out)
    pickle_out.close()
