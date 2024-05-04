
//image handle 

document.addEventListener('DOMContentLoaded', function() {
    const backButton = document.getElementById('backButton');
    backButton.addEventListener('click', () => {
        // Redirect the user back to the calendar page
        const urlParams = new URLSearchParams(window.location.search);
        const courseName = urlParams.get('courseName');
        window.location.href = `/`;
    });
});

document.getElementById('courseForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the form from submitting normally

    // Get form data
    let formData = new FormData();
    formData.append('course_name', document.getElementById('courseName').value);
    let images = document.getElementById('images').files;
    // for (let i = 0; i < images.length; i++) {
    formData.append('image', images[0]);
    // }

    // Check if confirmation checkbox is checked
    let confirmed = document.getElementById('confirmation').checked;

    // Send the data to the server
    if (confirmed) {
        fetch('/appload_image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response;
        })
        .then(data => {
            console.log('Upload successful:', data);
            alert('Upload successful!');
            // Optionally, display a success message or redirect to another page
        })
        .catch(error => {
            console.error('Upload error:', error);
        
        });
    } else {
        console.error('Please confirm the details before submitting.');
        // Optionally, display a message to the user indicating they need to confirm
    }
});

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
            day.dataset.date = `${year}-${month + 1}-${i - first_day.getDay() + 1}`; // Set the data-date attribute
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
document.addEventListener('DOMContentLoaded', function() {
    const viewAttendanceButton = document.getElementById('attendance-till-date-btn');
    const courseName = document.getElementById('courseHeading').textContent;
    viewAttendanceButton.addEventListener('click', function() {
        redirectToAttendancePage(courseName);
    });
});

function redirectToAttendancePage(courseName) {
    // Redirect to the attendance page with the 'source' parameter included
    window.location.href = `/attendance-page.html?source=button&courseName=${encodeURIComponent(courseName)}`;
}

//calendar color coding
document.addEventListener('DOMContentLoaded', function() {
    // Fetch attendance data and render the calendar
    fetchAttendanceData();
});

function fetchAttendanceData() {
    // Make a fetch request to get attendance data
    const urlParams = new URLSearchParams(window.location.search);
    const courseName = urlParams.get('courseName');
    fetch(`/calender-attendance?courseName=${encodeURIComponent(courseName)}`)
    
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Once the attendance data is fetched, render the calendar
            renderCalendar(data);
        })
        .catch(error => {
            console.error('Error fetching attendance data:', error);
        });
}

function checkVerified(attendance) {
    // Check if attendance is verified
    // You should implement your logic here based on your backend data structure
    // For example, if verified status is stored in the backend
    console.log("hk");
    return attendance.hasOwnProperty('verified') && attendance.verified === true;
}


function renderCalendar(attendanceData) {
    // Get the current date
    const currentDate = new Date();
   console.log(currentDate);
    // Loop through the days in the calendar
    const calendarDays = document.querySelectorAll('.calendar-day-hover');
    calendarDays.forEach(calendarDay => {
        // Parse the date from the data attribute of each calendar day
        const dateParts = calendarDay.dataset.date.split('-');
        const day = parseInt(dateParts[2]);
        const month = parseInt(dateParts[1]) - 1; // JavaScript months are zero-based
        const year = parseInt(dateParts[0]);
        const calendarDate = new Date(year, month, day);
        console.log(day);
        // Check if the calendar date is in the attendance data
        console.log(calendarDate);
        const formattedDate = formatDate(calendarDate);
        console.log(formattedDate);
        if (attendanceData[formattedDate] && attendanceData[formattedDate].verified) {
            console.log("hy");
            calendarDay.style.backgroundColor = '#2196F3'; // Blue for verified
        } else if (attendanceData[formattedDate]) {
            calendarDay.style.backgroundColor = '#4CAF50'; // Green for present but not verified
        } else {
            calendarDay.style.backgroundColor = '#FF5733'; // Red for absent
        }
        // Check if the calendar date is the current date
        if (calendarDate.toDateString() === currentDate.toDateString()) {
            calendarDay.classList.add('current-date');
        }
    });
}

function formatDate(date) {
    // Format the date as YYYY-MM-DD
    date.setDate(date.getDate() - 1);

    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}
