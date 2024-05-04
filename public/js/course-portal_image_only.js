// Script to fill course name automatically according to the heading of the page
// document.addEventListener('DOMContentLoaded', function() {
//     document.getElementById('courseName').value = document.querySelector('header h1').textContent;
// });

// // Script to open a date picker when clicking on the date input
// document.getElementById('date').addEventListener('click', function() {
//     this.type = 'date';
// });

//image handle 
document.getElementById('courseForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the form from submitting normally

    // Get form data
    let formData = new FormData();
    formData.append('course_name', document.getElementById('courseName').value);
    formData.append('date', document.getElementById('date').value);
    let images = document.getElementById('images').files;
    for (let i = 0; i < images.length; i++) {
        formData.append('images', images[i]); // Append each image to 'images'
    }

    // Check if attendance is already marked for the given date
    fetch('/api/check_attendance_date', {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${localStorage.getItem("idtoken")}`
        },
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.attendance_marked === 'yes') {
            const proceed = confirm('Attendance for this date is already marked. Do you still want to proceed?');
            if (proceed) {
                markAttendance(formData);
            }
        }
        else
        {
        markAttendance(formData);
        }
    })
    .catch(error => {
        console.error('Error checking attendance:', error);
        // Optionally, display an error message to the user
    });
});

function markAttendance(formData) {
    // Check if confirmation checkbox is checked
    let confirmed = document.getElementById('confirmation').checked;

    // Send the data to the server
    if (confirmed) {
        fetch('/api/appload_image', {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("idtoken")}`
            },
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            alert('Upload successful!');
            return response.json('Upload successful!');
        })
        .then(data => {
            console.log('Upload successful:', data);
            alert('Upload successful!');
            // Optionally, display a success message or redirect to another page
        })
        .catch(error => {
            console.error('Upload error:', error);
            // Optionally, display an error message to the user
        });
    } else {
        console.error('Please confirm the details before submitting.');
        // Optionally, display a message to the user indicating they need to confirm
    }
}


// document.getElementById('courseForm').addEventListener('submit', function(event) {
//     event.preventDefault(); // Prevent the form from submitting normally

//     // Get form data
//     let formData = new FormData();
//     formData.append('course_name', document.getElementById('courseName').value);
//     formData.append('date', document.getElementById('date').value);
//     // let images = document.getElementById('images').files;
//     // // for (let i = 0; i < images.length; i++) {
//     // formData.append('image', images[0]);
//     let images = document.getElementById('images').files;
//     for (let i = 0; i < images.length; i++) {
//         formData.append('images', images[i]); // Append each image to 'images'
//     }
//     // }

//     // Check if confirmation checkbox is checked
//     let confirmed = document.getElementById('confirmation').checked;

//     // Send the data to the server
//     if (confirmed) {
//         fetch('/api/appload_image', {
//             method: 'POST',
//             headers: {
//                 Authorization: `Bearer ${localStorage.getItem("idtoken")}`
//             },
//             body: formData
//         })
//         .then(response => {
//             if (!response.ok) {
//                 throw new Error('Network response was not ok');
//             }
//             alert('Upload successful!');
//             return response.json('Upload successful!');
//         })
//         .then(data => {
//             console.log('Upload successful:', data);
//             alert('Upload successful!');
//             // Optionally, display a success message or redirect to another page
//         })
//         .catch(error => {
//             console.error('Upload error:', error);
//             // Optionally, display an error message to the user
//         });
//     } else {
//         console.error('Please confirm the details before submitting.');
//         // Optionally, display a message to the user indicating they need to confirm
//     }
// });

// Course Name Population for course-portal.html page
document.addEventListener('DOMContentLoaded', function() {
    const courseHeading = document.getElementById('courseHeading');
    if (courseHeading) {
        const urlParams = new URLSearchParams(window.location.search);
        const courseName = urlParams.get('courseName');
        if (courseName) {
            courseHeading.textContent = courseName;
        }
    }
});

let calendar = document.querySelector('.calendar')

const month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

isLeapYear = (year) => {
    return (year % 4 === 0 && year % 100 !== 0 && year % 400 !== 0) || (year % 100 === 0 && year % 400 ===0)
}

getFebDays = (year) => {
    return isLeapYear(year) ? 29 : 28
}

generateCalendar = (month, year) => {

    let calendar_days = calendar.querySelector('.calendar-days')
    let calendar_header_year = calendar.querySelector('#year')

    let days_of_month = [31, getFebDays(year), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    calendar_days.innerHTML = ''

    let currDate = new Date()
    if (!month) month = currDate.getMonth()
    if (!year) year = currDate.getFullYear()

    let curr_month = `${month_names[month]}`
    month_picker.innerHTML = curr_month
    calendar_header_year.innerHTML = year

    // get first day of month
    
    let first_day = new Date(year, month, 1)

    for (let i = 0; i <= days_of_month[month] + first_day.getDay() - 1; i++) {
        let day = document.createElement('div')
        if (i >= first_day.getDay()) {
            day.classList.add('calendar-day-hover')
            day.innerHTML = i - first_day.getDay() + 1
            day.innerHTML += `<span></span>
                            <span></span>
                            <span></span>
                            <span></span>`
            if (i - first_day.getDay() + 1 === currDate.getDate() && year === currDate.getFullYear() && month === currDate.getMonth()) {
                day.classList.add('curr-date')
            }
        }
        calendar_days.appendChild(day)
        console.log("hy");
    }
}

let month_list = calendar.querySelector('.month-list')

month_names.forEach((e, index) => {
    let month = document.createElement('div')
    month.innerHTML = `<div data-month="${index}">${e}</div>`
    month.querySelector('div').onclick = () => {
        month_list.classList.remove('show')
        curr_month.value = index
        generateCalendar(index, curr_year.value)
    }
    month_list.appendChild(month)
})

let month_picker = calendar.querySelector('#month-picker')

month_picker.onclick = () => {
    month_list.classList.add('show')
}

let currDate = new Date()

let curr_month = {value: currDate.getMonth()}
let curr_year = {value: currDate.getFullYear()}

generateCalendar(curr_month.value, curr_year.value)

document.querySelector('#prev-year').onclick = () => {
    --curr_year.value
    generateCalendar(curr_month.value, curr_year.value)
}

document.querySelector('#next-year').onclick = () => {
    ++curr_year.value
    generateCalendar(curr_month.value, curr_year.value)
}
document.addEventListener('DOMContentLoaded', function() {
    const calendarDays = document.querySelector('.calendar-days');
    
    calendarDays.addEventListener('click', function(event) {
        const target = event.target;
        if (target.classList.contains('calendar-day-hover')) {
            const selectedDate = target.textContent;
            // const selectedMonth = month_names.indexOf(month_picker.textContent) + 1;
            const selectedMonthName = month_picker.textContent;
            const selectedMonth = (month_names.indexOf(selectedMonthName) + 1).toString().padStart(2, '0');
            const selectedYear = parseInt(calendar.querySelector('#year').textContent);
            const courseName = document.getElementById('courseHeading').textContent;
            console.log(courseName);
            //fetchAttendanceData(selectedYear, selectedMonth, selectedDate, courseName);
            navigateToAttendancePage(selectedYear, selectedMonth, selectedDate, courseName);
            
        }
    });
});
//for going to different page
function navigateToAttendancePage(year, month, date, courseName) {
    // Redirect to the attendance page with query parameters
    window.location.href = `/attendance-page.html?year=${year}&month=${month}&date=${date}&courseName=${encodeURIComponent(courseName)}`;
}

// function fetchAttendanceData(year, month, date, courseName) {
//     fetch(`/attendance?year=${year}&month=${month}&date=${date}&courseName=${courseName}`)
//         .then(response => {
//             if (!response.ok) {
//                 throw new Error('Network response was not ok');
//             }
//             return response.json();
//         })
//         .then(data => {
//             // Update UI to display attendance data
//             renderAttendanceData(data);
//             //console.log(data);
//             // Example: Render the data into a table or any other format
//         })
//         .catch(error => {
//             console.error('Error fetching attendance data:', error);
//             // Optionally, display an error message to the user
//         });
// }

// function renderAttendanceData(attendanceData) {
//     const attendanceContainer = document.getElementById('attendance-container');
//     attendanceContainer.innerHTML = ''; // Clear existing data

//     // Create an HTML table to display the attendance data
//     const table = document.createElement('table');
//     table.classList.add('attendance-table'); // Add styling if needed

//     // Create table header
//     const headerRow = document.createElement('tr');
//     headerRow.innerHTML = `
//         <th>Student ID</th>
//         <th>Attendance Status</th>
//         <th>pic1</th>
//         <th>pic2</th>
//         <th>pic3</th>
//         <th>pic4</th>
//         <th>pic5</th>
//         <!-- Add more columns if needed -->
//     `;
//     table.appendChild(headerRow);

//     // Create table body with attendance data
//     attendanceData.forEach(student => {
//         const row = document.createElement('tr');
//         row.innerHTML = `
//             <td>${student.student_id}</td>
//             <td>${student.p_or_a}</td>
//             <td><img src="${student.pic_1}" class="attendance-image" alt="Image 1"></td>
//             <td><img src="${student.pic_2}" class="attendance-image" alt="Image 2"></td>
//             <td><img src="${student.pic_3}" class="attendance-image" alt="Image 3"></td>
//             <td><img src="${student.pic_4}" class="attendance-image" alt="Image 4"></td>
//             <td><img src="${student.pic_5}" class="attendance-image" alt="Image 5"></td>
//             <!-- Add more columns if needed -->
//         `;
//         table.appendChild(row);
//     });

//     // Append the table to the container element
//     attendanceContainer.appendChild(table);

//     // Add a button or link to go back to the calendar page
//     const backButton = document.createElement('button');
//     backButton.textContent = 'Back to Calendar';
//     backButton.addEventListener('click', () => {
//         // Redirect the user back to the calendar page
//         const urlParams = new URLSearchParams(window.location.search);
//     const courseName = urlParams.get('courseName');
    
//     // Redirect the user back to the course-portal.html page with the course name parameter
//     window.location.href = `/course-portal.html?courseName=${encodeURIComponent(courseName)}`;
//     });
//     const link = document.createElement('link');
//         link.rel = 'stylesheet';
//         link.href = './css/attendance-page.css'; // Path to the attendance page CSS file
//         document.head.appendChild(link);
//     attendanceContainer.appendChild(backButton);

//     // Add CSS style to limit the size of the images
//     const style = document.createElement('style');
//     style.textContent = `
//         .attendance-image {
//             max-width: 100px; /* Adjust the maximum width as needed */
//             max-height: 100px; /* Adjust the maximum height as needed */
//             object-fit: cover; /* Maintain aspect ratio and crop as necessary */
//         }
//     `;
//     document.head.appendChild(style);
// }














  