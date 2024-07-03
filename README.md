# Smart Attendence System Using Face Recognition - Design Practicum, IIT Mandi

## Introduction

This project implements a Smart Attendance System using facial recognition technology, designed specifically for IIT Mandi. The system aims to automate the attendance tracking process, making it more efficient and accessible for both faculty and students.

Key features include:
- Centralized attendance portal accessible to all faculty and students
- Facial recognition for automated attendance marking
- Separate dashboards for faculty and students
- Comprehensive reporting and attendance management tools
- Integration with IIT Mandi's server infrastructure

The system is divided into four major components:
1. Model Specifications (using the face-recognition library)
2. Portal Design and Activation
3. Server and Database Design
4. Edge Device for Computation Optimization (Nvidia Jetson Orin Nano)

This solution minimizes cost while maximizing efficiency, providing a scalable attendance automation system for the institution.

## Detailed Information

For a comprehensive overview of the project, including technical details, system architecture, and implementation specifics, please refer to our full report:

[Link to Full Project Report](https://drive.google.com/file/d/1FEZNmfIlmHGEuCozcH84bnuqK_qp9C7k/view?usp=sharing)

## Setup Instructions

Follow these steps to set up and run the application:

### Prerequisites

- Node.js version v21.1.0 (or higher)

### Installation

1. **Install dependencies**

Run the following command in the project root directory:
```
npm install
```

2. **Set up Firebase credentials**

Save the `firebase-credentials.json` file to the project root folder. This file contains the necessary authentication information for Firebase services.

3. **Configure database credentials**

Open `app.js` and update the database credentials. Look for a section similar to this:

```javascript
const db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : 'your_password',
  database : 'your_database'
})
```
Replace the placeholder values with your actual database credentials.

4. **Start the application**

Run the following command to start the application:
```
node app.js
```





