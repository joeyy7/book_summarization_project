
1. install all the requirements
2. create a database and create a table as given below:
" 
CREATE TABLE books (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255),
    text TEXT,
    summary TEXT,
    key_aspects TEXT,
    important_points TEXT,
    sentiment_positive FLOAT,
    sentiment_neutral FLOAT,
    sentiment_negative FLOAT,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"
3. do the necessary changes eg. database password and username changes in the app.py code.