document.getElementById('student data').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the form from submitting normally

    // Get form data
    let formData = new FormData();
    // formData.append('student_name', document.getElementById('StudentName').value);
    let images = document.getElementById('images').files;
    // for (let i = 0; i < images.length; i++) {
    formData.append('image', images[0]);
    // }

    // Check if confirmation checkbox is checked
    let confirmed = document.getElementById('confirmation').checked;

    // Send the data to the server
    if (confirmed) {
        fetch('/api/student_upload_image', {
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
});


// router.post('/appload_image', upload.single('image'), async (req, res) => {
//     try {
//       const { course_name } = req.body;
//       const imageBuffer = req.file.buffer;
//       console.log("hi i am here");
//       // const { course_name, image} = req.body;
//       // const imageBuffer = Buffer.from(image, 'base64');
//       // fs.writeFileSync('uploaded_image.jpg', imageBuffer)
//           const response = await axios.post('http://127.0.0.1:5000/upload', {
//             image: imageBuffer.toString('base64')
//           })
//           .catch(function(error){
//             console.log(error.message)
//         });
//           console.log(response.data);
//           res.send(response.data);
//     } catch (error) {
//       console.log(error);
//       res.status(500).send(error);
//     }
//   });