from flask import Flask, render_template,request
import pickle

app = Flask(__name__)
filename = 'nlp_model.pkl'
model = pickle.load(open(filename,'rb'))
cv = pickle.load(open('transformer.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    vect = cv.transform(data)
    prediction = model.predict(vect)

    return render_template('result.html',prediction=prediction)


if __name__ == '__main__':
    app.run()
