from flask import Flask,render_template,request,jsonify
from src.pipeline.predict_pipeline import PredictPipeline,CustomData


application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("result.html")

    else:
        data = CustomData(
            age = int(request.form.get("age")),
            workclass = int(request.form.get("workclass")),
            education_num = int(request.form.get("education_num")),
            marital_status = int(request.form.get("marital_status")),
            occupation = int(request.form.get("occupation")),
            relationship = int(request.form.get("relationship")),
            race = int(request.form.get("race")),
            sex = int(request.form.get("sex")),
            capital_gain = int(request.form.get("capital_gain")),
            capital_loss = int(request.form.get("capital_loss")),
            hours_per_week = int(request.form.get("hours_per_week")),
            country = int(request.form.get("country")),
            )

        final_data = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_data)

        result = pred

        if result == 0:
            return render_template("result.html",final_result = "Your Income is Less Then Equal To 50K: {}".format(result))

        elif result == 1:
            return  render_template("result.html",final_result = "Your Income is More Then 50K: {}".format(result))


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)

