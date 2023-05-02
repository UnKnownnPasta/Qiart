

function triggerCmd() {
    document.getElementById("inp").style.display = "none";
    document.getElementById("canvas").style.display = "block";
    document.getElementById("wam").textContent = 'Hang in there.'
    document.getElementById("createBtn").style.display = "none";
    document.getElementById('canvas').style.display = 'flex'
    const canvas = new fabric.Canvas('canvas');

    const borderColor = document.getElementById("borderColor")
    var borderStyle = document.getElementById("borderRigidity")
    const bgColor = document.getElementById("bgColor")
    canvas.backgroundColor = bgColor.options[bgColor.selectedIndex].value;

    console.log(borderColor)

    borderColor.style.display = "none"
    borderStyle.style.display = "none"
    bgColor.style.display = "none"

    var rectWidth = 10;
    var canvasWidth = canvas.getWidth() - 1;
    var canvasHeight = canvas.getHeight() - 1;
    var rectColor = borderColor.options[borderColor.selectedIndex].value;

    borderStyle = borderStyle.options[borderStyle.selectedIndex].value;
    choice = {}
    solid = { fill: rectColor }
    dotted = {
        fill: 'rgba(0,0,0,0)',
        stroke: rectColor,
        strokeWidth: 1,
        strokeHeight: 1,
        strokeDashArray: [12, 0],
    }
    dashed = {
        fill: 'rgba(0,0,0,0)',
        stroke: rectColor,
        strokeWidth: 4,
        strokeHeight: 1,
        strokeDashArray: [10, 20],
    }

    if ( borderStyle == 'solid' ) { choice = solid }
    else if ( borderStyle == 'dotted' ) { choice = dotted }
    else if ( borderStyle == 'dashed' ) { choice = dashed }

    var rectTop = new fabric.Rect({
        left: 0, top: 0,
        width: canvasWidth, height: rectWidth,
        ...choice
    });
    var rectBottom = new fabric.Rect({
        left: 0, top: canvasHeight - rectWidth,
        width: canvasWidth, height: rectWidth,
        ...choice
    });
    var rectLeft = new fabric.Rect({
        left: 0, top: 0,
        width: rectWidth, height: canvasHeight,
        ...choice
    });
    var rectRight = new fabric.Rect({
        left: canvasWidth - rectWidth, top: 0,
        width: rectWidth, height: canvasHeight,
        ...choice
    });

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