const express = require("express");

const app = express()

app.get("/", (req,res) => {
    res.json("testing api");
})

const port = 9001

app.listen(port, () => console.log(`http://localhost:${port}`))
