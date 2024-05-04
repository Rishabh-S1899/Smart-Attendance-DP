document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.querySelector('.search-bar input');
    const courseList = document.querySelector('.course-list');
    const courses = document.querySelectorAll('.course');

    searchInput.addEventListener('input', function() {
        const searchTerm = searchInput.value.toLowerCase();

        courses.forEach(course => {
            const courseName = course.querySelector('h2').textContent.toLowerCase();
            if (courseName.includes(searchTerm)) {
                course.style.display = 'block';
            } else {
                course.style.display = 'none';
            }
        });
    });
});
