firebaseAuth.getAuth().onAuthStateChanged(function(user) {
  if (user) {
      console.log("user is logged in")
      console.log(user.email)
      let parts = user.email.split("@")[1];
      if(parts === 'students.iitmandi.ac.in')
      {
        window.location = "/student_front.html"
      }
      else if(user.email === 'vayungoel@gmail.com')
      {        
      }
      else if(parts !== 'iitmandi.ac.in')
      {
        alert("Use institute ID to login.");
        window.location = "/login.html";
      }
  } else {
    console.log("user not logged in")
    window.location = "/login.html";
}
});
  