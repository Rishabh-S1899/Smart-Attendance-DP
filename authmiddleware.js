const { getAuth } = require("firebase-admin/auth")
module.exports.authenticate = async (req, res, next) => {
  const header = req.headers['authorization'];
  console.log(header);
  if (!header) return res.status(401).json({ message: 'You are not not authorized.' })

  const [_, idtoken] = header.split(" ")
  console.log(idtoken)
  const auth = getAuth()
  auth
    .verifyIdToken(idtoken)
    .then((decodedtoken) => {
      auth.getUser(decodedtoken.uid)
        .then((userRecord) => {
          req.name = userRecord.displayName
          req.email = userRecord.email
          return next();
      })
    })
    .catch(err => {
      console.log('Token Verification failed because ', err.message)
      return res.status(500).json({ message: 'Invalid token. Please login again.' })
    })

}
