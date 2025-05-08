# FYP_TikiData
# TikiData Full-Stack Application

This guide explains how to set up and run both the backend (FastAPI) and frontend (React) components of the TikiData project.

---

## Prerequisites

* **Node.js & npm** (v14+ recommended)
* **Python** (v3.8+ recommended)
* **Git** (to clone the repository)

---

## 1. Clone the Repository

```bash
git clone https://github.com/RagibAnjum10/FYP-TikiData.git
cd FYP-TikiData
```

---

## 2. Backend Setup

1. Navigate to the backend folder:

   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:

   * **macOS/Linux:**  `python3 -m venv .venv && source .venv/bin/activate`
   * **Windows (PowerShell):**  `python -m venv .venv; .\.venv\Scripts\Activate`
3. Install Python dependencies:

   ```bash
   pip install -r requirments.txt
   ```
4. (Optional) Create a `.env` file if your app uses environment variables.
5. Start the FastAPI server:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
6. Open your browser to **[http://localhost:8000/docs](http://localhost:8000/docs)** to view the Swagger UI.

---

## 3. Frontend Setup

1. Open a new terminal and navigate to the frontend folder:

   ```bash
   cd ../frontend
   ```
2. Install JavaScript dependencies:

   ```bash
   npm install
   ```
3. Set your API base URL in a `.env` file at the frontend root:

   ```text
   REACT_APP_API_URL=http://localhost:8000/api
   ```
4. Start the React development server:

   ```bash
   npm start
   ```
5. Visit **[http://localhost:3000/](http://localhost:3000/)** in your browser to view the app.

---

## 4. Usage

* **Home**: Introduction to TikiData and project overview.
* **Make Prediction**: Navigate to the prediction view, select teams, and view model output.

---

## 5. Common Tasks

* **Add a new Python dependency**: `pip install <package>` and then `pip freeze > requirements.txt`.
* **Add a new JS dependency**: `npm install <package> --save`.
* **Rebuild frontend** (production): `npm run build`.

---

## 6. Troubleshooting

* If you see CORS errors, ensure your FastAPI `CORSMiddleware` allows origin `http://localhost:3000`.
* For blank pages, check browser console for React errors and verify the React server is running.
* To reset your environment:

  ````bash
  # Backend
  pip uninstall -r requirements.txt -y
  rm -rf .venv

  # Frontend
  rm -rf node_modules package-lock.json\ n```

  ````
