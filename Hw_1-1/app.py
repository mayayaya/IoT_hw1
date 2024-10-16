from flask import Flask, render_template, request, send_file
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

@app.route('/')
def index():
    # Default values for a, n, and c
    a = request.args.get('a', 1, type=float)
    n = request.args.get('n', 100, type=int)
    c = request.args.get('c', 10, type=float)

    return render_template('index.html', a=a, n=n, c=c)

@app.route('/plot')
def plot():
    # Get values from sliders
    a = request.args.get('a', 1, type=float)
    n = request.args.get('n', 100, type=int)
    c = request.args.get('c', 10, type=float)

    # Step 1: Generate random data
    variance = 10  # Variance of noise
    x = np.random.uniform(-10, 10, n)  # Generate n random x points between -10 and 10
    b = 50  # Intercept b fixed at 50

    # y = a*x + b + c*N(0, variance)
    y = a * x + b + c * np.random.normal(0, variance, n)

    # Step 2: Reshape the data for linear regression
    x = x.reshape(-1, 1)  # Reshape x to be a 2D array

    # Step 3: Apply Linear Regression
    model = LinearRegression()
    model.fit(x, y)

    # Get the slope and intercept from the model
    slope = model.coef_[0]
    intercept = model.intercept_

    # Step 4: Create points for the regression line
    x_line = np.linspace(-10, 10, 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    # Step 5: Plot the data and the regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, label='Data points', color='blue')  # Scatter plot of the data points
    ax.plot(x_line, y_line, label='Regression Line', color='red', linewidth=2)  # Plot regression line

    # Labeling
    ax.set_title(f'Linear Regression: y = {slope:.2f}x + {intercept:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)

    # Convert plot to PNG image and serve it as a response
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)  # Close the figure to avoid memory issues

    output.seek(0)
    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
