const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');
const fs = require('fs');
const axios = require('axios');
const multer  = require('multer');
const path = require('path');
const upload = multer();
const { authenticate } = require('./middleware')
const admin = require('firebase-admin')
const dotenv = require('dotenv')

dotenv.config()
admin.initializeApp()

function insertIntoSortedArray(sortedArray, [value, string, val2]) {
  const index = sortedArray.findIndex(([val, str]) => val > value);
  const insertIndex = index !== -1 ? index : sortedArray.length;
  sortedArray.splice(insertIndex, 0, [value, string, val2]);
  return sortedArray;
}

const db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : 'Vayun314',
  database : 'DP_test'
})

const app = express();
app.use(bodyParser.urlencoded({ extended: false }));
app.use(express.json({ limit: '10mb' }));

const router = express.Router();


db.connect(err =>{
    if(err) throw err;
    console.log('connected to db');
})



router.use(authenticate);

// router.get('/index',(req,res)=>{
//   console.log(req.query);
//   res.sendFile(path.join(__dirname, 'views', 'index.html'));
// });


// // Route to serve course-portal.html
// router.get('/course-portal.html', (req, res) => {
//   res.sendFile(path.join(__dirname, 'views', 'course-portal.html'));
// });


router.post('/verify_attendance' ,(req,res)=>{
  const { year, month, date, courseName } = req.query;
  // if (req.email.split("@")[1] !== 'iitmandi.ac.in') {
  //   return res.status(401).json({ message: 'User is not authenticated.'});
  // }
  let sql = `
      SELECT * FROM ${courseName}
      WHERE attendance_date = '${year}-${month}-${date}' ;
  `;
  db.query(sql,(err, results) => {
    if (err) {
        console.error('Error fetching attendance data:', err);
        res.status(500).json({ error: 'Internal Server Error' });
    } else {
        console.log(results);
        results.forEach(row => {
            let sql = `
              SELECT * FROM student
              WHERE student_id = '${row.student_id.toLowerCase()}' ;
            `;
            db.query(sql,(err,result)=>{
              if (err) {
                console.error('Error fetching attendance data:', err);
                res.status(500).json({ error: 'Internal Server Error' });
              } else {
                let original_vector;
                let alpha = 0.1;
                result.forEach(row1 => {
                original_vector = JSON.parse(row1.Student_vector);
                })
                if(row.pic_1 != null && row.pic_1 != '')
                {
                  let vec1 = JSON.parse(row.vector_1);
                  for (let i = 0; i < original_vector.length; i++) {
                  original_vector[i] = (1-alpha) * original_vector[i] + alpha * vec1[i];
                 }
                }
                if(row.pic_2 != null && row.pic_2 != '')
                {
                  let vec2 = JSON.parse(row.vector_2);
                  for (let i = 0; i < original_vector.length; i++) {
                  original_vector[i] = (1-alpha) * original_vector[i] + alpha * vec2[i];
                 }
                }
                if(row.pic_3 != null && row.pic_3 != '')
                {
                  let vec3 = JSON.parse(row.vector_3);
                  for (let i = 0; i < original_vector.length; i++) {
                  original_vector[i] = (1-alpha) * original_vector[i] + alpha * vec3[i];
                 }
                }
                if(row.pic_4 != null && row.pic_4 != '')
                {
                  let vec4 = JSON.parse(row.vector_4);
                  for (let i = 0; i < original_vector.length; i++) {
                  original_vector[i] = (1-alpha) * original_vector[i] + alpha * vec4[i];
                 }
                }
                if(row.pic_5 != null && row.pic_5 != '')
                {
                  let vec5 = JSON.parse(row.vetcor_5);
                  for (let i = 0; i < original_vector.length; i++) {
                  original_vector[i] = (1-alpha) * original_vector[i] + alpha * vec5[i];
                 }
                }
                const sql = `
                  UPDATE student
                  SET Student_vector = '${JSON.stringify(original_vector)}'
                  WHERE student_id = ?
                `;
                db.query(sql, [row.student_id.toLowerCase()], (err, result) => {
                  if (err) {
                      console.error('Error updating:', err);
                      res.status(500).json({ error: 'Internal Server Error' });
                  } else {
                      console.log('vector successfully updated:', result);
                      // res.json({ message: 'vector successfully updated' });
                  }
              });
            }
         })
      })
      res.json({ message: 'vector successfully updated' });
    }
});
})

router.post('/update_verify_column', (req,res) => {
  const { year, month, date, courseName } = req.query;
  let sql = `
      UPDATE ${courseName}
      SET verified = true
      WHERE attendance_date = '${year}-${month}-${date}' ;
  `;
  db.query(sql,(err, results) => {
    if (err) {
      console.error('Error fetching attendance data:', err);
      res.status(500).json({ error: 'Internal Server Error' });
    } else {
      res.json({ message: 'verified successfully' });
    }
  })
})



router.post('/update-attendance', (req, res) => {
  const { courseName, studentId, newStatus } = req.body;

  // Check if the course name, student ID, and new status are provided
  if (!courseName || !studentId || !newStatus) {
      return res.status(400).json({ error: 'Course name, student ID, and new status are required' });
  }

  // // Check if the new status is valid (P or A)
  // if (newStatus !== 'P' && newStatus !== 'A') {
  //     return res.status(400).json({ error: 'Invalid new status. It should be either P or A' });
  // }

  // Update the attendance status in the corresponding course table
  const sql = `
      UPDATE ${courseName} 
      SET p_or_a = ?
      WHERE student_id = ?
  `;
  db.query(sql, [newStatus, studentId], (err, result) => {
      if (err) {
          console.error('Error updating attendance status:', err);
          return res.status(500).json({ error: 'Internal Server Error' });
      }
      console.log('Attendance status updated successfully');
      res.status(200).json({ message: 'Attendance status updated successfully' });
  });
});

router.get('/attendance-all-dates', (req, res) => {
  const { courseName } = req.query;

  // Get today's date
  const today = new Date();
  const year = today.getFullYear();
  const month = (today.getMonth() + 1).toString().padStart(2, '0');
  const date = today.getDate().toString().padStart(2, '0');
  const currentDate = `${year}-${month}-${date}`;

  // Construct SQL query to fetch attendance data for all dates till today's date
  const sql = `
      SELECT student_id, attendance_date, p_or_a
      FROM ${courseName}
      WHERE attendance_date <= '${currentDate}'
  `;

  db.query(sql, (err, results) => {
      if (err) {
          console.error('Error fetching attendance data:', err);
          res.status(500).json({ error: 'Internal Server Error' });
      } else {
          
                    
            const attendanceData = results.reduce((acc, { student_id, attendance_date, p_or_a }) => {
              // Convert date to a string in YYYY-MM-DD format
              const formattedDate = attendance_date.toISOString().split('T')[0];
              
              // Initialize the date object if not exists
              acc[formattedDate] = acc[formattedDate] || {};
              // Add attendance data for the student
              acc[formattedDate][student_id] = p_or_a;
              
             
        
              return acc;
          }, {});
          console.log(attendanceData);
          res.json(attendanceData);
      }
  });
});


router.get('/attendance', (req, res) => {
  const { year, month, date, courseName } = req.query;

  // Modify this SQL query to fetch attendance data based on the provided parameters
  
  const sql = `
      SELECT * FROM ${courseName}
      WHERE attendance_date = '${year}-${month}-${date}' ;
  `;
  // console.log(sql);
  
  db.query(sql, [date], (err, results) => {
      if (err) {
          console.error('Error fetching attendance data:', err);
          res.status(500).json({ error: 'Internal Server Error' });
      } else {
          // console.log(results);
          res.json(results);
      }
  });
});

// router.post('/delete_image', (req, res) => {
//   const { courseName, studentId, imageNumber } = req.body;

//   // Construct the column name based on the image number
//   const columnName = `pic_${imageNumber}`;

//   // SQL query to update the image column to NULL
//   const sql = `
//       UPDATE ${courseName}
//       SET ${columnName} = NULL
//       WHERE student_id = ?
//   `;

//   // Execute the SQL query with the provided student ID
//   db.query(sql, [studentId], (err, result) => {
//       if (err) {
//           console.error('Error deleting image:', err);
//           res.status(500).json({ error: 'Internal Server Error' });
//       } else {
//           console.log('Image deleted successfully:', result);
//           res.json({ message: 'Image deleted successfully' });
//       }
//   });
// });

// router.post('/delete-image', (req, res) => {
//   const { courseName, studentId, imageNumber } = req.body;

//   // Construct the column name based on the image number
//   const columnName = `pic_${imageNumber}`;

//   // Start from the deleted column up to the second last column
//   for (let i = imageNumber; i < 5; i++) {
//     const currentColumnName = `pic_${i}`;
//     const nextColumnName = `pic_${i + 1}`;

//     // SQL query to fetch the value of the next column
//     const fetchNextImageSql = `
//         SELECT ${nextColumnName}
//         FROM ${courseName}
//         WHERE student_id = ?
//     `;

//     // Execute the SQL query to fetch the value of the next column
//     db.query(fetchNextImageSql, [studentId], (fetchErr, fetchResult) => {
//       if (fetchErr) {
//         console.error('Error fetching next image path:', fetchErr);
//         res.status(500).json({ error: 'Internal Server Error' });
//         return;
//       }

//       // Get the path of the next image from the fetched result
//       const nextImagePath = fetchResult.length > 0 ? fetchResult[0][nextColumnName] : null;

//       // SQL query to update the current column with the path of the next column
//       const updateSql = `
//           UPDATE ${courseName}
//           SET ${currentColumnName} = ?
//           WHERE student_id = ?
//       `;

//       // Execute the SQL query to update the current column with the path of the next column
//       db.query(updateSql, [nextImagePath, studentId], (updateErr, updateResult) => {
//         if (updateErr) {
//           console.error('Error updating image column:', updateErr);
//           res.status(500).json({ error: 'Internal Server Error' });
//         } else {
//           console.log('Image column updated successfully:', updateResult);
//         }
//       });
//     });
//   }

//   // Set the last column (pic_5) to NULL
//   const lastColumnName = 'pic_5';
//   const nullSql = `
//       UPDATE ${courseName}
//       SET ${lastColumnName} = NULL
//       WHERE student_id = ?
//   `;

//   // Execute the SQL query to set the last column to NULL
//   db.query(nullSql, [studentId], (nullErr, nullResult) => {
//     if (nullErr) {
//       console.error('Error setting last column to NULL:', nullErr);
//       res.status(500).json({ error: 'Internal Server Error' });
//     } else {
//       console.log('Last column set to NULL successfully:', nullResult);
//       res.json({ message: 'Image deleted successfully' });
//     }
//   });
// });


// router.post('/delete_image', (req, res) => {
//   const { courseName, studentId, imageNumber } = req.body;

//   // Construct the column name based on the image number
//   const columnName = `pic_${imageNumber}`;
//   console.log(`Image number is ${imageNumber}`)
//   // Start from the deleted column up to the second last column
//   for (let i = imageNumber; i < 5; i++) {
//     const currentColumnName = `pic_${i}`;
//     const nextColumnName = `pic_${i + 1}`;

//     // SQL query to fetch the value of the next column
//     const fetchNextImageSql = `
//         SELECT ${nextColumnName}
//         FROM ${courseName}
//         WHERE student_id = ?
//     `;

//     // Execute the SQL query to fetch the value of the next column
//     db.query(fetchNextImageSql, [studentId], (fetchErr, fetchResult) => {
//       if (fetchErr) {
//         console.error('Error fetching next image path:', fetchErr);
//         res.status(500).json({ error: 'Internal Server Error' });
//         return;
//       }
//       console.log("fetch",fetchResult);

//       // Get the path of the next image from the fetched result
//       const nextImagePath = fetchResult.length > 0 ? fetchResult[0][nextColumnName] : null;
//       console.log("img",nextImagePath);
//       // SQL query to update the current column with the path of the next column
//       const updateSql = `
//           UPDATE ${courseName}
//           SET ${currentColumnName} = ?
//           WHERE student_id = ?
//       `;

//       // Execute the SQL query to update the current column with the path of the next column
//       db.query(updateSql, [nextImagePath, studentId], (updateErr, updateResult) => {
//         if (updateErr) {
//           console.error('Error updating image column:', updateErr);
//           res.status(500).json({ error: 'Internal Server Error' });
//         } else {
//           console.log('Image column updated successfully:', updateResult);
//         }
//       });
//     });
//   }

//   // Set the last column (pic_5) to NULL
//   const lastColumnName = 'pic_5';
//   const nullSql = `
//       UPDATE ${courseName}
//       SET ${lastColumnName} = NULL
//       WHERE student_id = ?
//   `;

//   // Execute the SQL query to set the last column to NULL
//   db.query(nullSql, [studentId], (nullErr, nullResult) => {
//     if (nullErr) {
//       console.error('Error setting last column to NULL:', nullErr);
//       res.status(500).json({ error: 'Internal Server Error' });
//     } else {
//       console.log('Last column set to NULL successfully:', nullResult);
//       res.json({ message: 'Image deleted successfully' });
//     }
//   });
// });

//shru
// router.post('/delete_image', (req, res) => {
//   const { courseName, studentId, imageNumber,year,month,date } = req.body;

//   // Construct the column name based on the image number
//   const columnName = `pic_${imageNumber}`;
//   console.log(studentId);
//   // Start from the deleted column up to the second last column
//   for (let i = imageNumber; i < 5; i++) {
//     const currentColumnName = `pic_${i}`;
//     const nextColumnName = `pic_${i + 1}`;

//     // SQL query to fetch the value of the next column
//     const fetchNextImageSql = `
//         SELECT ${nextColumnName}
//         FROM ${courseName}
//         WHERE student_id = ? and attendance_date = '${year}-${month}-${date}'
//     `;
//     console.log(i,fetchNextImageSql);

//     // Execute the SQL query to fetch the value of the next column
//     db.query(fetchNextImageSql, [studentId], (fetchErr, fetchResult) => {
//       if (fetchErr) {
//         console.error('Error fetching next image path:', fetchErr);
//         res.status(500).json({ error: 'Internal Server Error' });
//         return;
//       }
//       console.log("fetch ",fetchResult);

//       // Get the path of the next image from the fetched result
//       const nextImagePath = fetchResult.length > 0 ? fetchResult[0][nextColumnName] : null;
//       console.log(`Next image path ${nextImagePath}`)
//       // SQL query to update the current column with the path of the next column
//       const updateSql = `
//           UPDATE ${courseName}
//           SET ${currentColumnName} = ?
//           WHERE student_id = ? and attendance_date = '${year}-${month}-${date}'
//       `;
//       console.log("update",updateSql);

//       // Execute the SQL query to update the current column with the path of the next column
//       db.query(updateSql, [nextImagePath, studentId], (updateErr, updateResult) => {
//         if (updateErr) {
//           console.error('Error updating image column:', updateErr);
//           res.status(500).json({ error: 'Internal Server Error' });
//         } else {
//           console.log('Image column updated successfully:', updateResult);
//         }
//       });
//     });
//   }

//   // Set the last column (pic_5) to NULL
//   const lastColumnName = 'pic_5';
//   const nullSql = `
//       UPDATE ${courseName}
//       SET ${lastColumnName} = NULL
//       WHERE student_id = ? and attendance_date = '${year}-${month}-${date}'
//   `;

//   // Execute the SQL query to set the last column to NULL
//   db.query(nullSql, [studentId], (nullErr, nullResult) => {
//     if (nullErr) {
//       console.error('Error setting last column to NULL:', nullErr);
//       res.status(500).json({ error: 'Internal Server Error' });
//     } else {
//       console.log('Last column set to NULL successfully:', nullResult);
//       res.json({ message: 'Image deleted successfully' });
// }
// });
// });


//Working code


// router.post('/delete_image', (req, res) => {
//   const { courseName, studentId, imageNumber, year, month, date } = req.body;

//   // Construct the column names based on the item number
//   const picColumnName = `pic_${imageNumber}`;
//   const vecColumnName = `vector_${imageNumber}`;
  
//   // Start from the deleted column up to the second last column
//   for (let i = imageNumber; i < 5; i++) {
//     const currentPicColumnName = `pic_${i}`;
//     const currentVecColumnName = `vector_${i}`;
//     const nextPicColumnName = `pic_${i + 1}`;
//     const nextVecColumnName = `vector_${i + 1}`;

//     // SQL query to fetch the values of the next image and vector columns
//     const fetchNextDataSql = `
//         SELECT ${nextPicColumnName}, ${nextVecColumnName}
//         FROM ${courseName}
//         WHERE student_id = ? and attendance_date = '${year}-${month}-${date}'
//     `;
    
//     // Execute the SQL query to fetch the values of the next image and vector columns
//     db.query(fetchNextDataSql, [studentId], (fetchErr, fetchResult) => {
//       if (fetchErr) {
//         console.error('Error fetching next data:', fetchErr);
//         res.status(500).json({ error: 'Internal Server Error' });
//         return;
//       }

//       // Get the values of the next image and vector from the fetched result
//       const nextImage = fetchResult.length > 0 ? fetchResult[0][nextPicColumnName] : null;
//       const nextVector = fetchResult.length > 0 ? fetchResult[0][nextVecColumnName] : null;

//       // SQL query to update the current image and vector columns with the values of the next columns
//       const updateSql = `
//           UPDATE ${courseName}
//           SET ${currentPicColumnName} = ?, ${currentVecColumnName} = ?
//           WHERE student_id = ? and attendance_date = '${year}-${month}-${date}'
//       `;

//       // Execute the SQL query to update the current image and vector columns with the values of the next columns
//       db.query(updateSql, [nextImage, nextVector, studentId], (updateErr, updateResult) => {
//         if (updateErr) {
//           console.error('Error updating columns:', updateErr);
//           res.status(500).json({ error: 'Internal Server Error' });
//         } else {
//           console.log('Columns updated successfully:', updateResult);
//         }
//       });
//     });
//   }

//   // Set the last columns (pic_5 and vec_5) to NULL
//   const lastPicColumnName = 'pic_5';
//   const lastVecColumnName = 'vector_5';
//   const nullSql = `
//       UPDATE ${courseName}
//       SET ${lastPicColumnName} = NULL, ${lastVecColumnName} = NULL
//       WHERE student_id = ? and attendance_date = '${year}-${month}-${date}'
//   `;

//   // Execute the SQL query to set the last columns to NULL
//   db.query(nullSql, [studentId], (nullErr, nullResult) => {
//     if (nullErr) {
//       console.error('Error setting last columns to NULL:', nullErr);
//       res.status(500).json({ error: 'Internal Server Error' });
//     } else {
//       console.log('Last columns set to NULL successfully:', nullResult);
      
//       res.json({ message: 'Image and vector deleted successfully' });
//     }
//   });
// });

//most -recent
router.post('/delete_image', (req, res) => {
  const { courseName, studentId, imageNumber, year, month, date } = req.body;

  // Construct the column name based on the image number
  const columnName = `pic_${imageNumber}`;

  // Start from the deleted column up to the second last column
  for (let i = imageNumber; i < 5; i++) {
    const currentColumnName = `pic_${i}`;
    const nextColumnName = `pic_${i + 1}`;

    // SQL query to fetch the value of the next column
    const fetchNextImageSql = `
      SELECT ${nextColumnName}
      FROM ${courseName}
      WHERE student_id = ? and attendance_date = '${year}-${month}-${date}'
    `;

    // Execute the SQL query to fetch the value of the next column
    db.query(fetchNextImageSql, [studentId], (fetchErr, fetchResult) => {
      if (fetchErr) {
        console.error('Error fetching next image path:', fetchErr);
        res.status(500).json({ error: 'Internal Server Error' });
        return;
      }

      // Get the path of the next image from the fetched result
      const nextImagePath = fetchResult.length > 0 ? fetchResult[0][nextColumnName] : null;

      // SQL query to update the current column with the path of the next column
      const updateSql = `
        UPDATE ${courseName}
        SET ${currentColumnName} = ?
        WHERE student_id = ? and attendance_date = '${year}-${month}-${date}'
      `;

      // Execute the SQL query to update the current column with the path of the next column
      db.query(updateSql, [nextImagePath, studentId], (updateErr, updateResult) => {
        if (updateErr) {
          console.error('Error updating image column:', updateErr);
          res.status(500).json({ error: 'Internal Server Error' });
        } else {
          console.log('Image column updated successfully:', updateResult);
        }
      });
    });
  }

  // Set the last column (pic_5) to NULL
  const lastColumnName = 'pic_5';
  const nullSql = `
    UPDATE ${courseName}
    SET ${lastColumnName} = NULL
    WHERE student_id = ?  and attendance_date = '${year}-${month}-${date}'
  `;

  // Execute the SQL query to set the last column to NULL
  db.query(nullSql, [studentId], (nullErr, nullResult) => {
    if (nullErr) {
      console.error('Error setting last column to NULL:', nullErr);
      res.status(500).json({ error: 'Internal Server Error' });
    } else {
      console.log('Last column set to NULL successfully:', nullResult);

      // Check if all images are null for the student on the given date
const checkAllImagesNullSql = `
SELECT 
  CASE 
    WHEN pic_1 IS NULL AND pic_2 IS NULL AND pic_3 IS NULL AND pic_4 IS NULL AND pic_5 IS NULL THEN 1 
    ELSE 0 
  END AS all_null
FROM ${courseName}
WHERE student_id = ? AND attendance_date = '${year}-${month}-${date}'
`;

// Execute the SQL query to check if all images are null
db.query(checkAllImagesNullSql, [studentId], (checkErr, checkResult) => {
if (checkErr) {
  console.error('Error checking if all images are null:', checkErr);
  res.status(500).json({ error: 'Internal Server Error' });
} else {
  const allNull = checkResult.length > 0 ? checkResult[0].all_null : 0;
  console.log('Check Result:', checkResult);
  console.log('All Null:', allNull);
  if (allNull === 1) { // All images are null
    // SQL query to update student_id to 'A'
    const updateStudentIdSql = `
      UPDATE ${courseName}
      SET p_or_a= 'A'
      WHERE student_id = ? AND attendance_date = '${year}-${month}-${date}'
    `;
    console.log('Update Student ID SQL:', updateStudentIdSql);
    // Execute the SQL query to update student_id to 'A'
    db.query(updateStudentIdSql, [studentId], (updateIdErr, updateIdResult) => {
      if (updateIdErr) {
        console.error('Error updating student_id:', updateIdErr);
        res.status(500).json({ error: 'Internal Server Error' });
      } else {
        console.log('Student ID updated to "A" for all null images:', updateIdResult);
        res.status(200).json({ message: 'Student ID updated to "A" for all null images' });
      }
    });
  } else {
    res.status(200).json({ message: 'Images updated successfully' });
  }
}
});

    }
  });
});






router.get('/professor', (req, res) => {
  // Extract professor ID from the query parameters
  // const professorId = req.query.professorId;  //fixing for now
  // if (!professorId) {
  //     return res.status(400).send('Professor ID is required');
  // }
  const professorId = req.email.split("@")[0];
  if (!professorId) {
      return res.status(400).send('Professor ID isrequired');
  }
  // Fetch courses corresponding to the professor ID from the database
  const sql = `SELECT * FROM professor_courses WHERE professor_id = ?`;
  db.query(sql, [professorId], (err, results) => {
      if (err) {
          console.error('Error fetching courses:', err);
          res.status(500).send('Internal Server Error');
      } else {
          // Send the fetched courses as JSON response
          res.json(results);
          console.log(results);
      }
  });
});

router.get('/search', (req, res) => {
  const searchTerm = req.query.q; // Get search term from query string
  console.log('Search term:', searchTerm); // Log the search term
  // const professorId=req.query.professorId;
  const professorId = req.email.split("@")[0];
  // SQL query to search for courses
  const sql = `SELECT * FROM professor_courses WHERE course_name LIKE '%${searchTerm}%' and professor_id='${professorId}'`;
  console.log('SQL query:', sql); // Log the SQL query

  db.query(sql, (err, results) => {
    if (err) {
      console.error('Error executing SQL query:', err);
      res.status(500).json({ error: 'Internal server error' });
      return;
    }
    console.log('Search results:', results); // Log the search results
    res.json(results); // Return search results as JSON
  });
});

router.get('/student_courses', (req, res) => {
  // const query = 'SELECT course_name FROM course_information';
  const query = 'SELECT * FROM course_information';
  db.query(query, (err, results) => {
    if (err) throw err;
    // console.log(res.json(results));
    console.log(results)
    res.json(results);
    
  });
});





router.post('/student_enroll', (req, res) => {
  const {courseName} = req.body;
  console.log(req.body);
  console.log(req.name);
  console.log(req.email);
  const studentId=req.email.slice(0,6)
  // const studentId = req.studentId;
  // const courseName = req.courseName;

  // console.log(courseName);
  console.log(studentId);

  // Check if the student is already enrolled in the selected course
  const checkQuery = 'SELECT * FROM student_enrolment WHERE student_id = ? AND course_id = ?';
  db.query(checkQuery, [studentId, courseName], (err, result) => {
    if (err) {
      console.error('Error checking enrolment:', err);
      res.status(500).json({ error: 'Error enrolling student' });
    } else if (result.length > 0) {
      // Student is already enrolled in the selected course
      res.status(200).json({ message: `You are already enrolled in the ${courseName} course` });
    } else {
      // Insert the selected course and student ID into the student_enrolment table
      const query = 'INSERT INTO student_enrolment (student_id, course_id) VALUES (?, ?)';
      db.query(query, [studentId, courseName], (err, result) => {
        if (err) {
          console.error('Error inserting data:', err);
          res.status(500).json({ error: 'Error enrolling student' });
        } else {
          // Send a success message to the client
          res.status(200).json({ message: `You have enrolled in the ${courseName} course` });
        }
      });
    }
  });
});


router.get('/student_courses_display', (req, res) => {
  const studentId = req.email.slice(0,6);
  console.log(studentId)
  if (!studentId) {
    return res.status(400).json({ error: 'Missing studentId' });
  }

  const query = `
  SELECT course_id from student_enrolment WHERE student_id = ?
  `;

  db.query(query, [studentId], (err, results) => {
    if (err) {
      console.error('Error fetching student courses:', err);
      return res.status(500).json({ error: 'Internal Server Error' });
    }

    console.log('Query results:', results);
    res.json(results);
  });
});

router.post('/delete_attendance_at_date', upload.array('images'), async (req, res) => {
  const { course_name, date } = req.body;
  console.log(`course name ${course_name}`)
  const sql = `DELETE FROM ${course_name} WHERE attendance_date = ?;`
  db.query(sql, [date], (error, results) => {
      if (error) {
          console.error('Error querying database:', error);
          res.status(500).json({ error: 'Internal server error' });
          return;
      }
      res.json({message:"previous attendence records deleted"})
});
})


router.post('/student_upload_image', upload.single('image'), async (req, res) => {
  try {
    const parts = req.name;
    let temp = parts.split("IIT Mandi");
    let studentname = temp[0].trim();
    const studentId=req.email.slice(0,6);
    // const studentId= 'b22222'
    // let studentname= 'Shruti Sahu'
    // let studentId = 'b22132';
    // console.log(extracted)
    const imageBuffer = req.file.buffer;
    console.log("hi i am here");
    console.log(studentname);
    console.log(imageBuffer);
    console.log(studentId);
    const response = await axios.post('http://127.0.0.1:5000/student_upload', {
      image: imageBuffer.toString('base64')
    })
    .catch(function(error) {
      console.log(error.message);
    });

    console.log(response.data);
    let currentDate = new Date();
    let currentYear = currentDate.getFullYear();
    let currentMonth = currentDate.getMonth() + 1; 
    let currentDay = currentDate.getDate();
    let currentHour = currentDate.getHours();
    let currentMinute = currentDate.getMinutes();
    let currentSecond = currentDate.getSeconds();
    let formattedDateTime = `${currentYear}-${currentMonth}_${currentDay}_${currentHour}_${currentMinute}_${currentSecond}`;
    fs.writeFileSync(`./public/images/${studentId}_${formattedDateTime}_ground_truth.jpg`, imageBuffer);
    let p = `public/images/${studentId}_${formattedDateTime}_ground_truth.jpg`;
    // Insert the data into the MySQL table
    db.query(
      'INSERT INTO student(student_id, Name_of_student, Student_vector, pic) VALUES (?, ?, ?, ?)',
      [studentId, studentname, JSON.stringify(response.data), p],
      (error, results, fields) => {
        if (error) {
          console.log(error);
          res.status(500).send(error);
        } else {
          console.log('Data Inserted Successfully');
          res.send(response.data);
        }
      }
    );
  } catch (error) {
    console.log(error);
    res.status(500).send(error);
  }
});

router.get('/unenrol_courses', (req, res) => {
  const studentId = req.email.slice(0,6);
  if (!studentId) {
    return res.status(400).json({ error: 'Missing studentId' });
  }

  const query = `
  SELECT * from student_enrolment WHERE student_id = ?
  `;
  db.query(query, [studentId], (err, results) => {
    if (err) {
      console.error('Error fetching student courses:', err);
      return res.status(500).json({ error: 'Internal Server Error' });
    }

    console.log('Query results:', results);
    res.json(results);
  });
});

router.post('/unenrol', (req, res) => {
  const { courseName } = req.body;
  const studentId=req.email.slice(0,6)
  db.query(
      'DELETE FROM student_enrolment WHERE student_id = ? AND course_id = ?',
      [studentId, courseName],
      (error, results) => {
          if (error) {
              console.error(error);
              res.status(500).send({ message: 'Error unenrolling from the course' });
          } else {
            console.log(results)
              if (results.affectedRows === 0) {
                  res.json({ message: 'You are not enrolled in this course' });
              } else {
                  res.json({ message: `You have unenrolled from ${courseName}` });
              }
          }
      }
  );
});

router.get('/attendance_for_student', (req, res) => {
  const courseName = req.query.courseName;
  // if (!courseName || courseName.trim() === '') {
  //   return res.status(400).json({ error: 'Invalid course name' });
  // }
  console.log(`This is coursename ${courseName}`)
  const tableName = `${courseName.replace(/\s+/g, '_').replace(/\W/g, '_')}`;
  const studentId=req.email.slice(0,6);
  console.log("This is the tablename")
  console.log(`This is the table name ${tableName}`)
  db.query(`SELECT  p_or_a, attendance_date FROM ${tableName} WHERE student_id= ?`, [studentId],(error, results) => {
      if (error) {
          console.error(error);
          res.status(500).json({ error: 'Error fetching attendance data' });
      } else {
        console.log("Results ",results)
          res.json(results);
      }
  });
});


router.get('/calender-attendance', (req, res) => {
  const { courseName } = req.query;

  // Get today's date
  const today = new Date();
  const year = today.getFullYear();
  const month = (today.getMonth() + 1).toString().padStart(2, '0');
  const date = today.getDate().toString().padStart(2, '0');
  const currentDate = `${year}-${month}-${date}`;

  // Construct SQL query to fetch attendance data for all dates till today's date
  const sql = `
      SELECT student_id, attendance_date, p_or_a,verified
      FROM ${courseName}
      WHERE attendance_date <= '${currentDate}'
  `;

  db.query(sql, (err, results) => {
      if (err) {
          console.error('Error fetching attendance data:', err);
          res.status(500).json({ error: 'Internal Server Error' });
      } else {
          
                    
            const attendanceData = results.reduce((acc, { student_id, attendance_date, p_or_a ,verified}) => {
              // Convert date to a string in YYYY-MM-DD format
              const formattedDate = attendance_date.toISOString().split('T')[0];
              
              // Initialize the date object if not exists
              acc[formattedDate] = acc[formattedDate] || {};
              
              // Add attendance data for the student
              acc[formattedDate][student_id] = p_or_a;
              // Add verified status
        acc[formattedDate].verified = verified === '1'; // Assuming 1 represents verified
              return acc;
          }, {});
          console.log(attendanceData);
          res.json(attendanceData);
      }
  });
});


// FACULTY SIDE ENDPOINTS
router.post('/check_appload',async (req,res)=>{
    try {
        const image = fs.readFileSync('images/pic1.jpeg');
        console.log("finished reading the file");
        const response = await axios.post('http://localhost:8000/appload_image', {
          course_name: "DP",
          image: image.toString('base64')
        });
        console.log(response.data);
        res.send(response.data);
      } 
      catch (error) {
        console.error(error);
    }
});

// router.post('/check_attendance_date', upload.single('video'), async (req, res) => {
router.post('/check_attendance_date', upload.array('images'), async (req, res) => {
  const { course_name, date } = req.body;
  console.log("Endpoint has been hit")
  const sql = `SELECT COUNT(*) AS count FROM ${course_name} WHERE attendance_date = ?`;
  db.query(sql, [date], (error, results) => {
      if (error) {
          console.error('Error querying database:', error);
          res.status(500).json({ error: 'Internal server error' });
          return;
      }
      // Check if there are any entries for the given date
      const count = results[0].count;
      console.log("i am checking the date")
      if (count > 0) {
          res.json({ attendance_marked: 'yes' });
      } else {
          res.json({ attendance_marked: 'no' });
      }
  });
})

// app.post('/appload_image',async (req,res)=>{
// router.post('/appload_video', upload.array('images'), async (req, res) => {
router.post('/appload_video', upload.single('video'), async (req, res) => {
  try {
    const { course_name, date } = req.body;
    const videoBuffer = req.file.buffer;
    // Process the video buffer here...
    console.log("hi i am here");
    console.log(date);
    const videoBase64 = videoBuffer.toString('base64');

    // Send the video to the specified endpoint
    const response = await axios.post('http://127.0.0.1:5000/upload_video', {
      video: videoBase64
    });
    // console.log(data);
    // const { course_name, image} = req.body;
    // // const imageBuffer = Buffer.from(image, 'base64');
    // // fs.writeFileSync('uploaded_image.jpg', imageBuffer)
    //     const response = await axios.post('http://127.0.0.1:5000/upload', {
    //       image: imageBuffer.toString('base64')
    //     })
    //     .catch(function(error){
    //       console.log(error.message)
    //     })
        console.log(response.data);
        let data  = response.data;
        // console.log(data)
      // const { course_name, date } = req.body;
      // const imageBuffer = req.file.buffer;
      // console.log("hi i am here");
      // console.log(date);
      // const images = req.files.map(file => file.buffer.toString('base64'));
      // let data = []; // Initialize an empty array to store responses
  
      // // Process each image buffer here...
      // for (let i = 0; i < images.length; i++) {
      //   const response = await axios.post('http://127.0.0.1:5000/upload', {
      //     image: images[i]
      //   });
      //   data = data.concat(response.data); // Append response to data array
      //   // console.log('Image uploaded:', response.data);
      // }
      // console.log(data);
      hashMap2 = {};
          
          // const sql = `SELECT * FROM student`
          const sql = `
          SELECT s.*
          FROM student s
          INNER JOIN student_enrolment se ON s.student_id = se.student_id
          WHERE se.course_id = "${course_name}"
          `
  
          db.query(sql, (err,result) =>{
              if(err) return console.log(err.message);
              console.log(result);
  
              let hashMap = {}
  
              result.forEach(row => {
                  hashMap[row.student_id.toLowerCase()] = JSON.parse(row.Student_vector);
                  hashMap2[row.student_id.toLowerCase()] = row.pic;
              })
              console.log(hashMap);
      
              let f;
              let k = [];
              for(let i= 0;i<data.length;i++)
              {
                  let min = Number.MAX_SAFE_INTEGER;
                  let min_id;
                  for (let key in hashMap) {
                      let sumOfSquares = 0;
                      list2 = hashMap[key]
                      for (let j = 0; j < list2.length; j++) {
                          sumOfSquares += Math.pow(data[i][1][j] - list2[j], 2);
                      }
                      f = Math.sqrt(sumOfSquares);
                      if(f < min)
                      {
                          min = f;
                          min_id = key;
                      }
                  }
                  k.push([min,min_id,data[i][0],data[i][1]])
              }
              
              const sql = `SELECT student_id FROM student_enrolment WHERE course_id='${course_name}'`
              data = k;
              hashMap = {};
  
             db.query(sql, (err,result) =>{
          if(err) return console.log(err.message);
          result.forEach(row => {
              hashMap[row.student_id] = [[Number.MAX_SAFE_INTEGER,"",[]],[Number.MAX_SAFE_INTEGER,"",[]],[Number.MAX_SAFE_INTEGER,"",[]],[Number.MAX_SAFE_INTEGER,"",[]],[Number.MAX_SAFE_INTEGER,"",[]]];
          })
          console.log(hashMap);
          console.log("here here")
          // data.sort((a, b) => b[0] - a[0]);
          console.log(data);
          for (let i = 0; i < data.length; i++) {
          let key = data[i][1];
          if (hashMap.hasOwnProperty(key)) 
              {
                  sortedList = insertIntoSortedArray(hashMap[key], [data[i][0],data[i][2],data[i][3]]);
                  if(sortedList.length > 5)
                  {
                      sortedList = sortedList.slice(0, 5);
                  }
                  hashMap[key] = sortedList;
                  // hashMap[key].push(data[i][0]); 
              }
          }
          let currentDate = new Date();
          let currentYear = currentDate.getFullYear();
          let currentMonth = currentDate.getMonth() + 1; // Months are zero-based
          let currentDay = currentDate.getDate();
          let currentHour = currentDate.getHours();
          let currentMinute = currentDate.getMinutes();
          let currentSecond = currentDate.getSeconds();
          for (const key in hashMap) {
            for(let i=0;i<hashMap[key].length;i++)
            {
              if(hashMap[key][i][1] != "")
              {
              let decodedImage = Buffer.from(hashMap[key][i][1], 'base64');
              let formattedDateTime = `${currentYear}-${currentMonth}_${currentDay}_${currentHour}_${currentMinute}_${currentSecond}`;
              fs.writeFileSync(`public/images/${key}_${formattedDateTime}_${i}.jpg`, decodedImage);
              hashMap[key][i][1] = `public/images/${key}_${formattedDateTime}_${i}.jpg`;
              }
            }
          }
          console.log(hashMap);
          for (const key in hashMap) {
              console.log("Entering the loop");
              let p_a = hashMap[key][0][0] === Number.MAX_SAFE_INTEGER ? 'A' : 'P';
              console.log([key, p_a, hashMap[key][0][0],hashMap[key][1][0],hashMap[key][2][0],hashMap[key][3][0],hashMap[key][4][0]]);
              const vector = JSON.stringify([hashMap[key][0][0],hashMap[key][1][0],hashMap[key][2][0],hashMap[key][3][0],hashMap[key][4][0]]); 
              const sql =
                `INSERT INTO ${course_name}
                (student_id, p_or_a, vector, pic_1, pic_2, pic_3, pic_4, pic_5, attendance_date, ground_truth, vector_1, vector_2,vector_3,vector_4,vetcor_5) VALUES
                ("${key}", "${p_a}", "${vector}", "${hashMap[key][0][1]}","${hashMap[key][1][1]}","${hashMap[key][2][1]}","${hashMap[key][3][1]}","${hashMap[key][4][1]}","${date}","${hashMap2[key]}","${JSON.stringify(hashMap[key][0][2])}","${JSON.stringify(hashMap[key][1][2])}","${JSON.stringify(hashMap[key][2][2])}","${JSON.stringify(hashMap[key][3][2])}","${JSON.stringify(hashMap[key][4][2])}");`; 
              db.query(sql, (err, result) => {
                if (err) return console.log(err.message);
                console.log(result);
              });
            } 
          // db.end();
          // res.send("hashMap");
      })
              // console.log(k);
              // try {
              //   const response = axios.post('/api/mark_attendence', {
              //     course_name : course_name,
              //     data : k
              //   });
              // }
              // catch(error){
              //   // console.error(error.message);
              // }
          })
  
          // face_images.forEach((encodedImage, index) => {
          //     const decodedImage = Buffer.from(encodedImage, 'base64');
          //     fs.writeFileSync(`face_image_${index}.jpg`, decodedImage);
          // });
        } 
        catch (error) {
          // console.error(error);
      }
      res.send("working")
  });


router.post('/appload_images', upload.array('images'), async (req, res) => {
  // router.post('/appload_image', upload.single('image'), async (req, res) => {
    try {
      const { course_name, date } = req.body;
      // const imageBuffer = req.file.buffer;
      console.log("hi i am here");
      console.log(date);
      console.log(course_name)
      const images = req.files.map(file => file.buffer.toString('base64'));
      let data = []; // Initialize an empty array to store responses
  
      // Process each image buffer here...
      for (let i = 0; i < images.length; i++) {
        const response = await axios.post('http://127.0.0.1:5000/upload', {
          image: images[i]
        });
        data = data.concat(response.data); // Append response to data array
        // console.log('Image uploaded:', response.data);
      }

      console.log(data);
      // const { course_name, image} = req.body;
      // const imageBuffer = Buffer.from(image, 'base64');
      // fs.writeFileSync('uploaded_image.jpg', imageBuffer)
          // const response = await axios.post('http://127.0.0.1:5000/upload', {
          //   image: imageBuffer.toString('base64')
          // })
          // .catch(function(error){
          //   console.log(error.message)
          // })
          // // console.log(response.data);
          // let data  = response.data;
          // console.log(data)
          hashMap2 = {};
          
          // const sql = `SELECT * FROM student`
          const sql = `
          SELECT s.*
          FROM student s
          INNER JOIN student_enrolment se ON s.student_id = se.student_id
          WHERE se.course_id = "${course_name}"
          `

  
          db.query(sql, (err,result) =>{
              if(err) return console.log(err.message);
              console.log(result);
  
              let hashMap = {}
  
              result.forEach(row => {
                  hashMap[row.student_id.toLowerCase()] = JSON.parse(row.Student_vector);
                  hashMap2[row.student_id.toLowerCase()] = row.pic;
              })
              console.log(hashMap);
      
              let f;
              let k = [];
              for(let i= 0;i<data.length;i++)
              {
                  let min = Number.MAX_SAFE_INTEGER;
                  let min_id;
                  for (let key in hashMap) {
                      let sumOfSquares = 0;
                      list2 = hashMap[key]
                      for (let j = 0; j < list2.length; j++) {
                          sumOfSquares += Math.pow(data[i][1][j] - list2[j], 2);
                      }
                      f = Math.sqrt(sumOfSquares);
                      if(f < min)
                      {
                          min = f;
                          min_id = key;
                      }
                  }
                  k.push([min,min_id,data[i][0],data[i][1]])
              }
              
              const sql = `SELECT student_id FROM student_enrolment WHERE course_id='${course_name}'`
              data = k;
              hashMap = {};
  
             db.query(sql, (err,result) =>{
          if(err) return console.log(err.message);
          result.forEach(row => {
              hashMap[row.student_id] = [[Number.MAX_SAFE_INTEGER,"",[]],[Number.MAX_SAFE_INTEGER,"",[]],[Number.MAX_SAFE_INTEGER,"",[]],[Number.MAX_SAFE_INTEGER,"",[]],[Number.MAX_SAFE_INTEGER,"",[]]];
          })
          console.log(hashMap);
          console.log("here here")
          // data.sort((a, b) => b[0] - a[0]);
          console.log(data);
          for (let i = 0; i < data.length; i++) {
          let key = data[i][1];
          if (hashMap.hasOwnProperty(key)) 
              {
                  sortedList = insertIntoSortedArray(hashMap[key], [data[i][0],data[i][2],data[i][3]]);
                  if(sortedList.length > 5)
                  {
                      sortedList = sortedList.slice(0, 5);
                  }
                  hashMap[key] = sortedList;
                  // hashMap[key].push(data[i][0]); 
              }
          }
          let currentDate = new Date();
          let currentYear = currentDate.getFullYear();
          let currentMonth = currentDate.getMonth() + 1; // Months are zero-based
          let currentDay = currentDate.getDate();
          let currentHour = currentDate.getHours();
          let currentMinute = currentDate.getMinutes();
          let currentSecond = currentDate.getSeconds();
          for (const key in hashMap) {
            for(let i=0;i<hashMap[key].length;i++)
            {
              if(hashMap[key][i][1] != "")
              {
              let decodedImage = Buffer.from(hashMap[key][i][1], 'base64');
              let formattedDateTime = `${currentYear}-${currentMonth}_${currentDay}_${currentHour}_${currentMinute}_${currentSecond}`;
              fs.writeFileSync(`public/images/${key}_${formattedDateTime}_${i}.jpg`, decodedImage);
              hashMap[key][i][1] = `public/images/${key}_${formattedDateTime}_${i}.jpg`;
              }
            }
          }
          console.log(hashMap);
          for (const key in hashMap) {
              console.log("Entering the loop");
              let p_a = hashMap[key][0][0] === Number.MAX_SAFE_INTEGER ? 'A' : 'P';
              console.log([key, p_a, hashMap[key][0][0],hashMap[key][1][0],hashMap[key][2][0],hashMap[key][3][0],hashMap[key][4][0]]);
              const vector = JSON.stringify([hashMap[key][0][0],hashMap[key][1][0],hashMap[key][2][0],hashMap[key][3][0],hashMap[key][4][0]]); 
              const sql =
                `INSERT INTO ${course_name}
                (student_id, p_or_a, vector, pic_1, pic_2, pic_3, pic_4, pic_5, attendance_date, ground_truth, vector_1, vector_2,vector_3,vector_4,vetcor_5) VALUES
                ("${key}", "${p_a}", "${vector}", "${hashMap[key][0][1]}","${hashMap[key][1][1]}","${hashMap[key][2][1]}","${hashMap[key][3][1]}","${hashMap[key][4][1]}","${date}","${hashMap2[key]}","${JSON.stringify(hashMap[key][0][2])}","${JSON.stringify(hashMap[key][1][2])}","${JSON.stringify(hashMap[key][2][2])}","${JSON.stringify(hashMap[key][3][2])}","${JSON.stringify(hashMap[key][4][2])}");`; 
              db.query(sql, (err, result) => {
                if (err) return console.log(err.message);
                console.log(result);
              });
            } 
          // db.end();
          // res.send("hashMap");
      })
              // console.log(k);
              // try {
              //   const response = axios.post('/api/mark_attendence', {
              //     course_name : course_name,
              //     data : k
              //   });
              // }
              // catch(error){
              //   // console.error(error.message);
              // }
          })
  
          // face_images.forEach((encodedImage, index) => {
          //     const decodedImage = Buffer.from(encodedImage, 'base64');
          //     fs.writeFileSync(`face_image_${index}.jpg`, decodedImage);
          // });
        } 
        catch (error) {
          // console.error(error);
      }
      res.send("working")
  });




router.post('/process_vector', (req,res)=>{
    const d1 = req.data;
    const sql = `SELECT * FROM student`
    const data = [
        [
          "/Users/vayungoel/Desktop/DP_BACKEND/face_0.jpg",
          [
            0.4904443621635437,
            0.12152142822742462,
            -0.030363233759999275,
            0.07516404986381531,
            -0.1515490561723709,
            -0.06912003457546234,
            -0.03220676630735397,
            0.5069826245307922,
            -0.18854951858520508,
            -0.06054352596402168
          ]
        ],
        [
          "/Users/vayungoel/Desktop/DP_BACKEND/face_1.jpg",
          [
            0.35696834325790405,
            0.24622885882854462,
            -0.13977298140525818,
            -0.16442148387432098,
            0.13163454830646515,
            -0.09940080344676971,
            -0.027319295331835747,
            0.13844461739063263,
            -0.18264994025230408,
            -0.01729181595146656
          ]
        ],
        [
          "/Users/vayungoel/Desktop/DP_BACKEND/face_2.jpg",
          [
            0.41893401741981506,
            0.033446744084358215,
            -0.11525203287601471,
            0.025625526905059814,
            -0.05647559463977814,
            -0.10904976725578308,
            0.06973914802074432,
            0.41553938388824463,
            0.11158615350723267,
            0.22551465034484863
          ]
        ],
        [
          "/Users/vayungoel/Desktop/DP_BACKEND/face_3.jpg",
          [
            0.37429624795913696,
            0.09435110539197922,
            -0.144717276096344,
            -0.00884789414703846,
            0.11659059673547745,
            -0.02606101520359516,
            0.011094517074525356,
            0.2098788321018219,
            0.09193816781044006,
            0.009681091643869877
          ]
        ],
        [
          "/Users/vayungoel/Desktop/DP_BACKEND/face_4.jpg",
          [
            0.34650853276252747,
            0.010625725612044334,
            0.03712175413966179,
            0.02375652827322483,
            -0.18990477919578552,
            -0.1550200879573822,
            -0.16256919503211975,
            0.20518165826797485,
            -0.12122160941362381,
            -0.171927347779274
          ]
        ],
        [
          "/Users/vayungoel/Desktop/DP_BACKEND/face_5.jpg",
          [
            0.17593923211097717,
            -0.12688593566417694,
            -0.03461556136608124,
            0.04746297746896744,
            -0.03174447640776634,
            0.08131656050682068,
            -0.09031643718481064,
            -0.12155752629041672,
            -0.06531347334384918,
            -0.06319945305585861
          ]
        ]
      ];

    let hashMap = {};

    db.query(sql, (err,result) =>{
        if(err) throw err;
        result.forEach(row => {
            hashMap[row.student_id] = JSON.parse(row.Student_vector);
        })
        console.log(hashMap);

        let f;
        let k = [];
        for(let i= 0;i<data.length;i++)
        {
            let min = Number.MAX_SAFE_INTEGER;
            let min_id;
            for (let key in hashMap) {
                let sumOfSquares = 0;
                list2 = hashMap[key]
                for (let j = 0; j < list2.length; j++) {
                    sumOfSquares += Math.pow(data[i][1][j] - list2[j], 2);
                }
                f = Math.sqrt(sumOfSquares);
                if(f < min)
                {
                    min = f;
                    min_id = key;
                }
            }
            k.push([min,min_id,data[i][0]])
        }
        console.log(k);
    })

    res.send("hashMap");

})

// for marking attendence based on confidence score list
router.post('/mark_attendence', (req,res)=> {
    const { course_name, data } = req.body;
    const sql = `SELECT student_id FROM student_enrolment WHERE course_id='${course_name}'`

    let hashMap = {};

    db.query(sql, (err,result) =>{
        if(err) return console.log(err.message);
        result.forEach(row => {
            hashMap[row.student_id] = [[Number.MAX_SAFE_INTEGER,""],[Number.MAX_SAFE_INTEGER,""],[Number.MAX_SAFE_INTEGER,""],[Number.MAX_SAFE_INTEGER,""],[Number.MAX_SAFE_INTEGER,""]];
        })
        console.log(hashMap);
        // data.sort((a, b) => b[0] - a[0]);
        // console.log(data);
        for (let i = 0; i < data.length; i++) {
        let key = data[i][1];
        if (hashMap.hasOwnProperty(key)) 
            {
                sortedList = insertIntoSortedArray(hashMap[key], [data[i][0],data[i][2]]);
                if(sortedList.length > 5)
                {
                    sortedList = sortedList.slice(0, 5);
                }
                hashMap[key] = sortedList;
                // hashMap[key].push(data[i][0]); 
            }
        }
        console.log(hashMap);
        for (const key in hashMap) {
            console.log("Entering the loop");
            let p_a = hashMap[key][0][0] === Number.MAX_SAFE_INTEGER ? 0 : 1;
            console.log([key, p_a, hashMap[key][0][0],hashMap[key][1][0],hashMap[key][2][0],hashMap[key][3][0],hashMap[key][4][0]]);
            const vector = JSON.stringify([hashMap[key][0][0],hashMap[key][1][0],hashMap[key][2][0],hashMap[key][3][0],hashMap[key][4][0]]); 
            const sql =
              `INSERT INTO DP_2024_summer 
              (student_id, p_or_a, vector, pic_1, pic_2, pic_3, pic_4, pic_5) VALUES
              ("${key}", "${p_a}", "${vector}", "${hashMap[key][0][1]}","${hashMap[key][1][1]}","${hashMap[key][2][1]}","${hashMap[key][3][1]}","${hashMap[key][4][1]}");`; 
            db.query(sql, (err, result) => {
              if (err) return console.log(err.message);
              console.log(result);
            });
          } 
        
        db.end();
        res.send("hashMap");
    })
    

    
})

// to add a new course attendence table for every new course
router.post('/add_new_course', (req, res) => {
    const { course_name, year, semester } = req.body;
    console.log(`Received form data: Course Name - ${course_name}, Year - ${year}, Semester - ${semester}`);
    const sql = `
    CREATE TABLE ${course_name}_${year}_${semester} (
    student_id VARCHAR(255),
    p_or_a VARCHAR(1),
    vector VARCHAR(255),
    FOREIGN KEY (student_id) REFERENCES student(student_id)
  )
`;
    db.query(sql, (err,result) =>{
    if(err) throw err;
    console.log(result);
    })

    res.send(`Received form data: Course Name - ${course_name}, Year - ${year}, Semester - ${semester}`);
});

//create a new database
router.get('/createdb', (req,res) => {
    let sql = 'CREATE DATABASE samplesql';
    db.query(sql, (err,result) => {
        if(err) throw err;
        res.send('Database created');
    });
});

app.use('/api', router )

app.use(express.static('public'));
app.use(express.static('views'));

app.listen('8000', () => {
    console.log('server started');
})

