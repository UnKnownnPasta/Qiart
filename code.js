function triggerCmd() {
    document.getElementById("inp").style.display = "none";
    document.getElementById("borderColor").style.display = "none";
    document.getElementById("borderRigidity").style.display = "none";
    document.getElementById("bgColor").style.display = "none";
    document.getElementById("canvas").style.display = "block";
    document.getElementsByTagName("h1")[0].textContent = 'Hang in there.'
    document.getElementsByTagName("button")[0].style.display = "none";

    document.getElementById('canvas').style.display = 'flex'
    const canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = 'gray'

    var rectWidth = 10;
    var canvasWidth = canvas.getWidth() - 1;
    var canvasHeight = canvas.getHeight() - 1;
    var rectColor = 'black';
    var rectTop = new fabric.Rect({ left: 0, top: 0, width: canvasWidth, height: rectWidth, fill: rectColor });
    var rectBottom = new fabric.Rect({ left: 0, top: canvasHeight - rectWidth, width: canvasWidth, height: rectWidth, fill: rectColor });
    var rectLeft = new fabric.Rect({ left: 0, top: 0, width: rectWidth, height: canvasHeight, fill: rectColor });
    var rectRight = new fabric.Rect({ left: canvasWidth - rectWidth, top: 0, width: rectWidth, height: canvasHeight, fill: rectColor });
    canvas.add(rectTop, rectBottom, rectLeft, rectRight);

    var inputText = document.getElementById('inp').value
    console.log(inputText)
    var input = "Give me some poetic info on " + inputText + ' in 20 words. You:'
    console.log(input)
    fetch('https://api.openai.com/v1/engines/text-davinci-002/completions', {
        method: 'POST',
        headers: {
            'Authorization': 'Bearer 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',
            'Content-Type': 'application/json',
            'temperature': 0.1
        },
        body: JSON.stringify({
            prompt: input,
            max_tokens: 100
        })
    })
        .then(async response => await response.json())
        .then(data => {
            console.log(data)
            document.getElementById('h2').innerText = data.choices[0].text;
            document.getElementById('h2').style.display = 'block'
            document.getElementsByTagName('h1')[0].textContent = ''
        })
        .catch(error => console.error(error));

}