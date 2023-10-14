from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


@app.route('/', methods=["get", "post"])
def predict():
    message = ""
    if request.method == "POST":
        IW = request.form.get("IW")
        IF = request.form.get("IF")
        VW = request.form.get("VW")
        FP = request.form.get("FP")

        # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        # model_loaded.predict([[3., 1., 29., 0., 0., 8.]])

        parms = [[float(IW), float(IF), float(VW), float(FP)]]

        with open('knn_best.pkl', 'rb') as file:
            knn_best = pickle.load(file)

        with open('knn_neighbors.pkl', 'rb') as file:
            knn_neighbors = pickle.load(file)

        y_width = knn_best.predict(parms)
        y_depth = knn_neighbors.predict(parms)

        message = f"Ширина сварного шва: {y_width[0][0]}; Глубина сварного шва: {y_depth[0][0]} "

    return render_template("index.html", message=message)

if __name__ == '__main__':
    app.run()
