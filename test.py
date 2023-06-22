import roboflow

rf = roboflow.Roboflow(api_key="7ryBC8sKb0QeK9S2EXmK")

# get a project
project = rf.workspace().project("upc-3sj4d/ara_si")

# Retrieve the model of a specific project
model = project.version("1").model

# predict on a local image
#prediction = model.predict("./TinyTLP/Jet3/img/00101.jpg")
prediction = model.predict("./image.jpg")
print(prediction)
prediction.save(output_path='predictions.jpg')
