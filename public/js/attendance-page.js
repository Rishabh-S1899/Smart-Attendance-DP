document.getElementById('newButton').addEventListener('click', function() {
    const urlParams = new URLSearchParams(window.location.search);
    const year = urlParams.get('year');
    const month = urlParams.get('month');
    const date = urlParams.get('date');
    const courseName = urlParams.get('courseName');
    var confirmDelete = confirm('Are you sure you want to proceed?');
    if (confirmDelete) {
        // Proceed with the action
        fetch(`/api/verify_attendance?year=${year}&month=${month}&date=${date}&courseName=${courseName}`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("idtoken")}`
            }
        })
        .then(response => {
            // Handle the API response, e.g., check for success status
            if (response.ok) {
                // Call another function that makes a second API call
                return verifying_column(year, month, date, courseName);
            } else {
                // Handle the API error, e.g., show an error message
                console.error('Failed to make API call');
                alert('You are not authenticated.');
            }
        })
        // .then(response => {
        //     if (response.ok) {
        //         return response.json();
        //     } else {
        //         throw new Error('Failed to authenticate user.');
        //     }
        // })
        // .then(data => {
        //     // Check if the response contains a message indicating that the user is not authenticated
        //     if (data.message && data.message === 'User is not authenticated.') {
        //         alert('You are not authenticated.');
        //     } else {
        //         // Call another function that makes a second API call
        //         return verifying_column(year, month, date, courseName);
        //     }
        // })
        .then(secondApiResponse => {
            // Handle the response from the second API call
            // This block will only be executed if the first API call was successful
            console.log('Second API call response:', secondApiResponse);
            window.location.href = '/index.html';
        })
        .catch(error => {
            console.error('Error making API call:', error);
        });
    } else {
        // Do nothing if the user cancels
        console.log('User canceled.');
    }
});


function verifying_column(year, month, date, courseName) {
    // Make the second API call here
    return fetch(`/api/update_verify_column?year=${year}&month=${month}&date=${date}&courseName=${courseName}`, {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${localStorage.getItem("idtoken")}`
        }
    })
    .then(response => {
        // Return the response for further processing in the next .then() block
        return response.json();
    })
    .catch(error => {
        console.error('Error making second API call:', error);
    });
}


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

document.addEventListener('DOMContentLoaded', function() {
    const backButton = document.getElementById('backButton');
    backButton.addEventListener('click', () => {
        // Redirect the user back to the calendar page
        const urlParams = new URLSearchParams(window.location.search);
        const courseName = urlParams.get('courseName');
        window.location.href = `/course-portal.html?courseName=${encodeURIComponent(courseName)}`;
    });

    // Fetch attendance data and render it
    // Check if the URL has a specific parameter to determine the source
    const urlParams = new URLSearchParams(window.location.search);
    const source = urlParams.get('source');

    if (source === 'button') {
        // If the source is the button click, fetch and render attendance data for the button view
        fetchAllAttendanceData();
    } else {
        // If the source is not specified or it's from the calendar, fetch and render attendance data as usual
        fetchAttendanceData();
    }
    
});
function fetchAllAttendanceData(){
    console.log("all");
    const urlParams = new URLSearchParams(window.location.search);
    const courseName = urlParams.get('courseName');
    fetch(`/api/attendance-all-dates?courseName=${encodeURIComponent(courseName)}`,{
        headers: {
            Authorization: `Bearer ${localStorage.getItem("idtoken")}`
        }
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            renderAllAttendanceData(data);
        })
        .catch(error => {
            console.error('Error fetching attendance data:', error);
        });
}


function renderAllAttendanceData(attendanceData) {
    const attendanceContainer = document.getElementById('attendance-container');
    attendanceContainer.innerHTML = ''; // Clear existing data
    var button = document.getElementById("newButton");

    // Hide the button by setting its display property to "none"
    if (button) {
        button.style.display = "none";
    }


    const table = document.createElement('table');
    table.classList.add('attendance-table');

    
    // Get all dates and sort them in ascending order
    const dates = Object.keys(attendanceData).sort((a, b) => new Date(a) - new Date(b));

    // Create table header
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th>Student ID</th>';

    // Dynamically create table header for each date
    dates.forEach(date => {
        const formattedDate = new Date(date).toLocaleDateString('en-GB'); // Format date as dd/mm/yyyy
        headerRow.innerHTML += `<th>${formattedDate}</th>`;
    });
    // Add additional columns
    headerRow.innerHTML += '<th>Total Present</th>';
    headerRow.innerHTML += '<th>Total Absent</th>';
    headerRow.innerHTML += '<th>Percentage Attendance</th>';
    headerRow.innerHTML += '<th>Total Days of Classes</th>';

    table.appendChild(headerRow);


    // Create table body with attendance data
    const studentIds = Object.keys(attendanceData[dates[1]]); // Get student IDs from the first date
    studentIds.forEach(studentId => {
        const row = document.createElement('tr');
        let totalPresent = 0;
        let totalAbsent = 0;

        row.innerHTML = `<td>${studentId}</td>`;
        dates.forEach(date => {
            const attendanceStatus = attendanceData[date][studentId] || '-';
            row.innerHTML += `<td>${attendanceStatus}</td>`;
            if (attendanceStatus === 'P') {
                totalPresent++;
            } else if (attendanceStatus === 'A') {
                totalAbsent++;
            }
        });
        // Calculate percentage attendance
        const totalDaysOfClass = dates.length;
        
        const percentageAttendance = ((totalPresent / (totalPresent+totalAbsent)) * 100).toFixed(2);
        if(totalAbsent+totalPresent==0){
            const percentageAttendance=0;
        }

        // Populate additional columns
        row.innerHTML += `<td>${totalPresent}</td>`;
        row.innerHTML += `<td>${totalAbsent}</td>`;
        row.innerHTML += `<td>${percentageAttendance}%</td>`;
        row.innerHTML += `<td>${totalDaysOfClass}</td>`;

        table.appendChild(row);
    });


    attendanceContainer.appendChild(table);
}




function fetchAttendanceData() {
    // Extract query parameters from URL
    const urlParams = new URLSearchParams(window.location.search);
    const year = urlParams.get('year');
    const month = urlParams.get('month');
    const date = urlParams.get('date');
    const courseName = urlParams.get('courseName');

    // Make fetch request to get attendance data
    fetch(`/api/attendance?year=${year}&month=${month}&date=${date}&courseName=${courseName}`,{
        headers: {
            Authorization: `Bearer ${localStorage.getItem("idtoken")}`
        }
      })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            renderAttendanceData(data);
        })
        .catch(error => {
            console.error('Error fetching attendance data:', error);
        });
        // Get all delete buttons
        const deleteButtons = document.querySelectorAll('.delete-button');  
}

function renderAttendanceData(attendanceData) {
//     const attendanceContainer = document.getElementById('attendance-container');
//     attendanceContainer.innerHTML = ''; // Clear existing data

//     // Create an HTML table to display the attendance data
//     const table = document.createElement('table');
//     table.classList.add('attendance-table');

//     // Create table header
//     const headerRow = document.createElement('tr');
//     headerRow.innerHTML = `
//         <th>Student ID</th>

//         <th>Attendance Status</th>
//         <th>real_pic</th>
//         <th>Picture 1</th>
//         <th>Picture 1</th>
//         <th>Picture 2</th>
//         <th>pic3</th>
//        <th>pic4</th>
//        <th>pic5</th>
//         <!-- Add more picture columns if needed -->
//     `;
//     table.appendChild(headerRow);

//     // Create table body with attendance data
//     attendanceData.forEach(student => {
//         const row = document.createElement('tr');
//         row.innerHTML = `
//             <td>${student.student_id}</td>
            
// <td>
//     <select class="attendance-status" data-student-id="${student.student_id}">
//     <option value="P" ${student.p_or_a === 'P' || student.p_or_a === '1' ? 'selected' : ''}>P</option>
//     <option value="A" ${student.p_or_a === 'A' || student.p_or_a === '0' ? 'selected' : ''}>A</option>
//     </select>
// </td>   
// <td>
//     ${student.real_pic ? `<img src="${student.real_pic.slice(6)}" class="attendance-image" alt="Image 1">` : 'Deleted'}
// </td>


// <td>
// ${student.pic_1 ? `<img src="${student.pic_1.slice(6)}" class="attendance-image" alt="Image 1"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 1)">Delete</button>` : 'Deleted'}
// </td>
// <td>
// ${student.pic_2 ? `<img src="${student.pic_2.slice(6)}" class="attendance-image" alt="Image 2"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 2)">Delete</button>` : 'Deleted'}
// </td>
// <td>
// ${student.pic_3 ? `<img src="${student.pic_3.slice(6)}" class="attendance-image" alt="Image 3"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 3)">Delete</button>` : 'Deleted'}
// </td>
// <td>
// ${student.pic_4 ? `<img src="${student.pic_4.slice(6)}" class="attendance-image" alt="Image 4"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 4)">Delete</button>` : 'Deleted'}
// </td>
// <td>
// ${student.pic_5 ? `<img src="${student.pic_5.slice(6)}" class="attendance-image" alt="Image 5"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 5)">Delete</button>` : 'Deleted'}
// </td>
//             <!-- Add more picture columns if needed -->
//         `;
//         table.appendChild(row);

        
    
//     });

//     attendanceContainer.appendChild(table);
//         // Add event listener to handle changes in attendance status
//     const attendanceInputs = document.querySelectorAll('.attendance-status');
//     attendanceInputs.forEach(input => {
//     input.addEventListener('change', handleAttendanceStatusChange);

    
// });

    const attendanceContainer = document.getElementById('attendance-container');
        attendanceContainer.innerHTML = ''; // Clear existing data

        // Create an HTML table to display the attendance data
        const table = document.createElement('table');
        table.classList.add('attendance-table');

        // Create table header
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `
            <th>Student ID</th>
            <th>Attendance Status</th>
            <th>real_pic</th>
            <th>Picture 1</th>
            <th>Picture 2</th>
            <th>Picture 3</th>
        <th>Picture 4</th>
        <th>Picture 5</th>
            <!-- Add more picture columns if needed -->
        `;
        table.appendChild(headerRow);

        // Create table body with attendance data
        attendanceData.forEach(student => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${student.student_id}</td>
                
    <td>
        <select class="attendance-status" data-student-id="${student.student_id}">
        <option value="P" ${student.p_or_a === 'P' || student.p_or_a === '1' ? 'selected' : ''}>P</option>
        <option value="A" ${student.p_or_a === 'A' || student.p_or_a === '0' ? 'selected' : ''}>A</option>
        <option value="L" ${student.p_or_a === 'L' || student.p_or_a === 'L' ? 'selected' : ''}>L</option>
        </select>
    </td>   
    <td>
        ${student.real_pic ? `<img src="${student.real_pic.slice(6)}" class="attendance-image" alt="Image 1">` : 'Deleted'}
    </td>


    <td>
    ${student.pic_1 ? `<img src="${student.pic_1.slice(6)}" class="attendance-image" alt="Image 1"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 1)">Delete</button>` : 'Deleted'}
    </td>
    <td>
    ${student.pic_2 ? `<img src="${student.pic_2.slice(6)}" class="attendance-image" alt="Image 2"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 2)">Delete</button>` : 'Deleted'}
    </td>
    <td>
    ${student.pic_3 ? `<img src="${student.pic_3.slice(6)}" class="attendance-image" alt="Image 3"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 3)">Delete</button>` : 'Deleted'}
    </td>
    <td>
    ${student.pic_4 ? `<img src="${student.pic_4.slice(6)}" class="attendance-image" alt="Image 4"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 4)">Delete</button>` : 'Deleted'}
    </td>
    <td>
    ${student.pic_5 ? `<img src="${student.pic_5.slice(6)}" class="attendance-image" alt="Image 5"><button class="delete-image-btn" onclick="handleImageDeletion('${student.student_id}', 5)">Delete</button>` : 'Deleted'}
    </td>
                <!-- Add more picture columns if needed -->
            `;
            table.appendChild(row);

            
        
        });

        attendanceContainer.appendChild(table);
            // Add event listener to handle changes in attendance status
            const verifyButton = document.getElementById('newButton');
        verifyButton.textContent = 'Verify';
    
        // Add Verify button CSS dynamically
        const verifyButtonCss = document.createElement('style');
        verifyButtonCss.textContent = `
            #newButton {
                position: absolute;
                top: 15px;
                right: 15px;
                padding: 8px 16px;
                font-size: 16px;
                background-color: #007bff;
                color: #fff;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
    
            #newButton:hover {
                background-color: #0056b3;
            }
        `;
        document.head.appendChild(verifyButtonCss);
        const attendanceInputs = document.querySelectorAll('.attendance-status');
        attendanceInputs.forEach(input => {
        input.addEventListener('change', handleAttendanceStatusChange);

        
    });

function handleAttendanceStatusChange(event) {
    const input = event.target;
    const newStatus = input.value;
    const studentId = input.dataset.studentId;
    console.log("sd");
    // Show confirmation dialog
    const confirmed = confirm(`Are you sure you want to change the attendance status to '${newStatus}' for student ${studentId}?`);
    if (confirmed) {
        // Send updated attendance status to the backend
        updateAttendanceStatus(studentId, newStatus);
    }
    else {
        // Reset the input value if the user cancels the operation
        // input.value = getAttendanceStatus(studentId); // You need to implement this function
        location.reload();
    }
}

function updateAttendanceStatus(studentId, newStatus) {
    // Make a fetch request to update attendance status
    // Get course name from the heading
    const courseName = document.getElementById('courseHeading').textContent;
    fetch('/api/update-attendance', {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${localStorage.getItem("idtoken")}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            courseName: courseName,
            studentId: studentId,
            newStatus: newStatus
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Attendance status updated successfully:', data);
        // Optionally, update UI to reflect the changes
        // const attendanceStatusElement = document.querySelector(`.attendance-status[data-student-id="${studentId}"]`);
        // if (attendanceStatusElement) {
        //     attendanceStatusElement.value = newStatus;
        // }
    })
    .catch(error => {
        console.error('Error updating attendance status:', error);
        // Optionally, display an error message to the user
    });
}


    
    
    //     // Add a button or link to go back to the calendar page
    // const backButton = document.createElement('button');
    // backButton.textContent = 'Back to Calendar';
    // backButton.addEventListener('click', () => {
    //     // Redirect the user back to the calendar page
    //     const urlParams = new URLSearchParams(window.location.search);
    // const courseName = urlParams.get('courseName');
    
    // Redirect the user back to the course-portal.html page with the course name parameter
    // window.location.href = `/course-portal.html?courseName=${encodeURIComponent(courseName)}`;
    // });
    const link = document.createElement('link');
        // link.rel = 'stylesheet';
        // link.href = './css/attendance-page.css'; // Path to the attendance page CSS file
        document.head.appendChild(link);
    // attendanceContainer.appendChild(backButton);
    const style = document.createElement('style');
    style.textContent = `
        .attendance-image {
            max-width: 100px; /* Adjust the maximum width as needed */
            max-height: 100px; /* Adjust the maximum height as needed */
            object-fit: cover; /* Maintain aspect ratio and crop as necessary */
        }
    `;
    document.head.appendChild(style);


}

//delete image
// Add event listener to handle image deletion
function handleImageDeletion(studentId, imageNumber) {
    const confirmed = confirm(`Are you sure you want to delete this image?`);
    if (confirmed) {
        // Send request to backend to delete image
        deleteImage(studentId, imageNumber);
    }
}

//Function to delete image
function deleteImage(studentId, imageNumber) {
    const courseName = document.getElementById('courseHeading').textContent;
    const urlParams = new URLSearchParams(window.location.search);
    const year = urlParams.get('year');
    const month = urlParams.get('month');
    const date = urlParams.get('date');

    // Send POST request to delete image
    fetch('/api/delete_image', {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${localStorage.getItem("idtoken")}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            courseName: courseName,
            studentId: studentId,
            imageNumber: imageNumber,
            year: year,
            month: month,
            date: date
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Image deleted successfully:', data);
        // Optionally, update UI to reflect the changes
        // For example, remove the deleted image from the DOM
        // Remove the image element from the DOM
        fetch('/api/update_attendance_status', {
            method: 'POST',
            headers: {
              Authorization: `Bearer ${localStorage.getItem("idtoken")}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              courseName: courseName,
              studentId: studentId,
              year: year,
              month: month,
              date: date
            })
          });
        
        location.reload();
    })
    .catch(error => {
        console.error('Error deleting image:', error);
        // Optionally, display an error message to the user
    });
}

