from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

filename = 'model\Hackathonmodel2.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        UserName = request.form['UserName']
        Department = request.form['Department']
        Redmine = request.form['Redmine']
        DocPortral = request.form['DocPortral']
        InternalFileSharing = request.form['InternalFileSharing']
        FTPServer = request.form['FTPServer']
        Server = request.form['Server']
        Firewall = request.form['Firewall']
        ConFile = request.form['ConFile']
        result = np.array([[Redmine, DocPortral, InternalFileSharing, FTPServer, Server, Firewall, ConFile]])

        prediction = model.predict(result)
        if prediction < 0.57:
            display = UserName+" from "+Department+" Department has all correct Access as per IGA"
            print(display)
        elif prediction > 0.57:
            display = UserName+" from "+Department+" Department has Wrong Access !!! please link it with IGA"
            print(display)

    return render_template("submit.html", n=display)


if __name__ == "__main__":
    app.run(debug = True)
