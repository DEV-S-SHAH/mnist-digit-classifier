"""
Simple HTTP server that loads the trained MNIST model (mnist.keras)
and provides a /predict endpoint for the drawing web app (predict.html).

Usage:
    python server.py

The server runs on http://localhost:5757
Open predict.html in your browser after starting this server.
"""

import json
import base64
import io
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

# Load the model once at startup
import tensorflow as tf
print("Loading model...")
model = tf.keras.models.load_model("mnist.keras")
print("Model loaded successfully!")


class PredictHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for health checks and digit prediction."""

    def _set_headers(self, status=200, content_type="application/json"):
        """Send response headers with CORS support."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        # Allow requests from any origin (needed since HTML is opened as a file)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self._set_headers(200)

    def do_GET(self):
        """Handle GET requests (serve HTML page and health check)."""
        if self.path == "/" or self.path == "/predict.html":
            # Serve the drawing web app
            import os
            html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predict.html")
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    html = f.read()
                self._set_headers(200, content_type="text/html")
                self.wfile.write(html.encode("utf-8"))
            except FileNotFoundError:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "predict.html not found"}).encode())
        elif self.path == "/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "not found"}).encode())

    def do_POST(self):
        """Handle POST requests (prediction endpoint)."""
        if self.path != "/predict":
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "not found"}).encode())
            return

        try:
            # Read the request body
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            # Decode the base64 PNG image from the canvas
            image_b64 = data["image"].split(",")[1]  # strip "data:image/png;base64,"
            image_bytes = base64.b64decode(image_b64)

            # Convert to a PIL image, resize to 28x28, convert to grayscale
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("L")
            img = img.resize((28, 28), Image.LANCZOS)

            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Apply the same tf.keras.utils.normalize the training notebook used
            img_array = tf.keras.utils.normalize(img_array)

            # Reshape to match model input: (1, 28, 28)
            img_input = img_array.reshape(1, 28, 28)

            # Run prediction
            predictions = model.predict(img_input, verbose=0)
            probs = predictions[0].tolist()
            predicted_digit = int(np.argmax(probs))

            # Create a small preview of the processed 28x28 image (before normalize)
            # Re-read original grayscale for preview
            preview_img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28), Image.LANCZOS)
            buf = io.BytesIO()
            preview_img.save(buf, format="PNG")
            preview_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Send the result
            self._set_headers(200)
            result = {
                "prediction": predicted_digit,
                "probabilities": probs,
                "processed_image": preview_b64,
            }
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[server] {args[0]}")


if __name__ == "__main__":
    PORT = 5757
    server = HTTPServer(("localhost", PORT), PredictHandler)
    print(f"Server running at http://localhost:{PORT}")
    print(f"Open http://localhost:{PORT} in your browser to start drawing!")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
