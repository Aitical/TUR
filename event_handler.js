document.addEventListener('DOMContentLoaded', domReady);

function domReady() {
    // new Dics({
    //     container: document.querySelectorAll('.b-dics')[0],
    //     hideTexts: false,
    //     textPosition: "bottom"

    // });
    // new Dics({
    //     container: document.querySelectorAll('.b-dics')[1],
    //     hideTexts: false,
    //     textPosition: "bottom"

    // });
    new Dics({
        container: document.querySelector('.b-dics'),
        hideTexts: false,
        textPosition: "bottom"
    });
}

function realworld3DEvent(idx) {
    let dics = document.querySelectorAll('.b-dics')[0]
    let sections = dics.getElementsByClassName('b-dics__section')
    let imagesLength = 5
    for (let i = 0; i < imagesLength; i++) {
        let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
        switch (idx) {
            case 0:
                image.src = 'images/realworld/mart0006';
                break;
            case 1:
                image.src = 'images/realworld/sushi0009';
                break;
            case 2:
                image.src = 'images/realworld/izakaya0003';
                break;
            case 3:
                image.src = 'images/realworld/park0000';
                break; 
            case 4:
                image.src = 'images/realworld/subway0000';
                break;
        }
        switch (i) {
            case 0:
                image.src = image.src + '_nerfw.png';
                break;
            case 1:
                image.src = image.src + '_denerf.png';
                break;
            case 2:
                image.src = image.src + '_wildgs.png';
                break;
            case 3:
                image.src = image.src + '_gsw.png';
                break;
            case 4:
                image.src = image.src + '_our.png';
                break;
        }
    }

    let scene_list = document.getElementById("realworld-3d-filter").children;
    for (let i = 0; i < scene_list.length; i++) {
        if (idx == i) {
            scene_list[i].children[0].className = "nav-link active"
        }
        else {
            scene_list[i].children[0].className = "nav-link"
        }
    }
}