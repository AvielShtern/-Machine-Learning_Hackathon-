Police departments around the world are building machine learning systems, which attempt to predict
time, place and type of crimes. With an effective learning system, the police might be able to
direct patrols to certain locations at certain times that are more prone to occurrence of a crime, and
hopefully prevent it from happening in the first place. we receive a dataset of 35,000 crimes that
happened in Chicago. A sample contains location of the crime (x,y), time, type, and more.

Challenges:
In this task there is a primary challenge and a secondary challenge. The primary challenge is
classification of crime type: we build a learning system that, given crime feature
vectors predicts which kind of crime it is, from 5 classes.
The secondary challenge is crime prevention. Every day we can send 30 police cars to the city - we direct a car to
a specific location and a specific time of day. If a crime was about to happen up to 500m from the
location you specified and up to 30 minutes before or after the time we specified, we prevented a
crime! we build a learning system that, given a date in the future, will output 30
(x,y,time) combinations - where time is during that day from midnight to midnight. Police cars will
be sent to these locations at these times.


the files:
classifier.py - given by staff
main.py - runs and creates the models(used only for training)
Preproccesor.py - Preproccess the data and trains the classifier model
TrainModelForSendPoliceCars.py - Trains the Police misson model
TrainedSerializedModel.bin - pickel serialized model