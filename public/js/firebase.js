  // Import the functions you need from the SDKs you need
  import { initializeApp } from "https://www.gstatic.com/firebasejs/10.10.0/firebase-app.js";
  import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.10.0/firebase-analytics.js";
  import * as firebaseAuth from 'https://www.gstatic.com/firebasejs/10.10.0/firebase-auth.js'

  // https://firebase.google.com/docs/web/setup#available-libraries

  // Your web app's Firebase configuration
  // For Firebase JS SDK v7.20.0 and later, measurementId is optional
  const firebaseConfig = {
    apiKey: "AIzaSyD1IaFavQj8PGKe2N5MU_ytENVmTpdcpkw",
    authDomain: "dp-2024-57a19.firebaseapp.com",
    projectId: "dp-2024-57a19",
    storageBucket: "dp-2024-57a19.appspot.com",
    messagingSenderId: "865391959568",
    appId: "1:865391959568:web:a1e585477d0f73a304b495",
    measurementId: "G-P7CLFTK434"
  };

  // Initialize Firebase
  const app = initializeApp(firebaseConfig);
  const analytics = getAnalytics(app);

  window.app = app
  window.firebaseAuth = firebaseAuth
  
  // TODO: Add SDKs for Firebase products that you want to use
