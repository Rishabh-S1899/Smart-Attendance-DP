// document.addEventListener('DOMContentLoaded', function() {
//     const searchInput = document.querySelector('.search-bar input');
//     const courseList = document.querySelector('.course-list');
//     const courses = document.querySelectorAll('.course');

//     searchInput.addEventListener('input', function() {
//         const searchTerm = searchInput.value.toLowerCase();

//         courses.forEach(course => {
//             const courseName = course.querySelector('h2').textContent.toLowerCase();
//             if (courseName.includes(searchTerm)) {
//                 course.style.display = 'block';
//             } else {
//                 course.style.display = 'none';
//             }
//         });
//     });
// });



document.addEventListener('DOMContentLoaded', function() {
  // Function to fetch courses and render them on the page
  console.log("hy");
  const fetchAndRenderCourses = function() {
      // Make a GET request to fetch courses for the specified professor ID
      // console.log(professorId);
      fetch(`/api/professor/`,{
        headers: {
            Authorization: `Bearer ${localStorage.getItem("idtoken")}`
        }
      })
          .then(response => response.json())
          .then(courses => {
              // Select the course list container
              const courseList = document.querySelector('.course-list');

              // Clear existing courses
              courseList.innerHTML = '';
              console.log(courseList);
              // Render each course
              courses.forEach(course => {
                  const courseElem = document.createElement('div');
                  courseElem.classList.add('course');
                  courseElem.innerHTML = `
                  <h2><a href="course-portal.html?courseName=${course.course_name}">${course.course_name}</a></h2>
                      <p>Course Slots: ${course.slots}</p>
                      
                  `;
                  courseList.appendChild(courseElem);
              });
          })
          .catch(error => console.log('Error fetching courses:', error));
  };

  // Fetch and render courses when the page loads
  fetchAndRenderCourses(); // Replace 1 with the actual professor ID

  
});


 
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.querySelector('.search-bar input');
    const searchIcon = document.querySelector('.search-bar .bx-search-alt-2');
    const courseList = document.querySelector('.course-list');
  
    // Function to perform search
    const performSearch = function() {
      const searchTerm = searchInput.value.trim();
      console.log('Search term:', searchTerm); // Log the search term
      
      // Make AJAX request to backend
      fetch(`/api/search?q=${searchTerm}`,{
        headers: {
            Authorization: `Bearer ${localStorage.getItem("idtoken")}`
        }
      })
        .then(response => response.json())
        .then(data => {
          console.log('Search results:', data); // Log the search results
  
          courseList.innerHTML = ''; // Clear previous results
  
          if (data.length === 0) {
            courseList.innerHTML = '<p>No results found</p>';
          } else {
            data.forEach(course => {
              const courseElem = document.createElement('div');
              courseElem.classList.add('course');
              courseElem.innerHTML = `
              <h2><a href="course-portal.html?courseName=${course.course_name}">${course.course_name}</a></h2>
                <p>Course Slot: ${course.course_slot}</p>
              `;
              courseList.appendChild(courseElem);
            });
          }
        })
        .catch(error => console.error('Error fetching data:', error));
    };
  
    // Event listener for clicking the search icon
    searchIcon.addEventListener('click', performSearch);
  
    // Event listener for pressing the Enter key
    searchInput.addEventListener('keypress', function(event) {
      if (event.key === 'Enter') {
        performSearch();
      }
    });
    // Code to populate course name on course portal page
    const urlParams = new URLSearchParams(window.location.search);
    const courseName = urlParams.get('courseName');
    if (courseName) {
        document.getElementById('courseHeading').textContent = courseName;
    }
  });

  // on clicking any course

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