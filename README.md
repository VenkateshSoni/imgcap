# imgcap

**imgcap** is an image captioning application leveraging the [LLaVA](https://llava-vl.github.io/) (Large Language and Vision Assistant) model to generate descriptive captions for images. This project integrates advanced vision-language models to provide accurate and context-aware image descriptions.

## Features

- **Automated Image Captioning**: Generates descriptive captions for input images using the LLaVA model.
- **Web Interface**: User-friendly web interface for uploading images and viewing captions.
- **Extensible Architecture**: Modular design allowing easy integration with other models or expansion of functionalities.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/VenkateshSoni/imgcap.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd imgcap
   ```

3. **Install Required Dependencies**:

   Ensure you have Python installed. Then, install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application**:

   ```bash
   python app.py
   ```

   This will start the web server.

2. **Access the Web Interface**:

   Open your web browser and navigate to `http://localhost:5000`. Here, you can upload images and receive generated captions.

## Project Structure

- `static/`: Contains static files like CSS, JavaScript, and images.
- `templates/`: Holds HTML templates for the web interface.
- `app.py`: Main application script initializing and running the web server.
- `requirements.txt`: Lists all Python dependencies required for the project.

## Contributing

Contributions are welcome! If you'd like to improve this project, please fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---
