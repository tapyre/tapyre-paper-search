# Installation Guide

![Install Guide Logo](images/logo-install-guide.png)

---

## Production Installation

1. **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2. **Build and Start Services**
    ```bash
    docker compose build
    docker compose up -d
    ```

3. **Access the Application**
    - Open [http://localhost:8000](http://localhost:8000) in your browser.

4. **Monitoring and Logs**
    ```bash
    docker compose logs -f
    ```

---

## Development Installation

### 1. Update Source Code

- In `mysql_database.py`, **uncomment**:
    ```python
    self.db_url = "mysql+pymysql://root:root@127.0.0.1:3306/test"
    ```
- **Comment out** the following lines:
    ```python
    # user = os.getenv("MYSQL_USER")
    # password = os.getenv("MYSQL_PASSWORD")
    # host = os.getenv("MYSQL_HOST", "mysql_db")
    # database = os.getenv("MYSQL_DATABASE")
    # self.logger.debug("[MySQLDatabase] Env vars - USER: %s, HOST: %s, DB: %s", user, host, database)
    # if not all([user, password, database]):
    #     raise ValueError("[MySQLDatabase] Missing env vars: MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")
    # self.db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"
    ```

- In `qdrant_database.py`, **change**:
    ```python
    self.host = os.getenv("QDRANT_HOST", "qdrant_db")
    ```
    **to**:
    ```python
    self.host = os.getenv("QDRANT_HOST", "localhost")
    ```

---

### 2. Start Databases with Docker

- **Qdrant DB**:
    ```bash
    docker run -d \
      --name qdrant_db \
      -p 6333:6333 \
      -p 6334:6334 \
      -e QDRANT__STORAGE__PATH="/qdrant/storage" \
      -v $(pwd)/qdrant_storage:/qdrant/storage \
      qdrant/qdrant
    ```

- **MySQL DB**:
    ```bash
    docker run -d \
      --name mysql_db \
      -p 3306:3306 \
      -e MYSQL_ROOT_PASSWORD=root \
      -e MYSQL_DATABASE=test \
      -v $(pwd)/mysql_data:/var/lib/mysql \
      mysql:8.0
    ```

---

### 3. Run the Application

- **Start the API**:
    ```bash
    uv run src/api.py
    ```

- **Start the Pipeline**:
    ```bash
    uv run src/pipeline.py
    ```

---

### 4. Verify

- Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to check if the API is running.