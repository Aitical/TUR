function playMergeVid(vid, videoMerge) {
  // Written by Dor Verbin, October 2021
  // This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
  // With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/
  // With additional modifications by Jonas Kulhanek 2024
  const leftcaption = [...videoMerge.parentElement.querySelectorAll('.video-label > :first-child')];
  const rightcaption = [...videoMerge.parentElement.querySelectorAll('.video-label > :last-child')];
  const borderThickness = 4;
  var position = 0.5;
  var elementWidth = 0;
  var vidWidth = vid.videoWidth/2;
  var vidHeight = vid.videoHeight;
  var bcr = videoMerge.getBoundingClientRect();
  var mergeContext = videoMerge.getContext("2d");

  function updateCaptionMasks() {
    if (leftcaption) {
      for (var i = 0; i < leftcaption.length; i++) {
        leftcaption[i].style.clipPath = `xywh(0 0 ${bcr.width*position-borderThickness/2}px 100%)`;
      }
    }
    if (rightcaption) {
      for (var i = 0; i < rightcaption.length; i++) {
        rightcaption[i].style.clipPath = `inset(0 0 0 calc(100% - ${bcr.width * (1-position)-borderThickness/2}px))`;
      }
    }
  }

  updateCaptionMasks();

  if (vid.readyState > 3) {
    vid.play();

    function trackLocation(e) {
      // Normalize to [0, 1]
      bcr = videoMerge.getBoundingClientRect();
      position = ((e.pageX - bcr.x) / bcr.width);
      updateCaptionMasks();
    }
    function trackLocationTouch(e) {
      // Normalize to [0, 1]
      bcr = videoMerge.getBoundingClientRect();
      position = ((e.touches[0].pageX - bcr.x) / bcr.width);
      updateCaptionMasks();
    }

      videoMerge.addEventListener("mousemove",  trackLocation, false);
      videoMerge.addEventListener("touchstart", trackLocationTouch, false);
      videoMerge.addEventListener("touchmove",  trackLocationTouch, false);

      function clamp(number, min, max) {
        return Math.min(Math.max(number, min), max);
      };

      function drawLoop() {
          mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
          var colStart = clamp(vidWidth * position, 0.0, vidWidth);
          var colWidth = clamp(vidWidth - (vidWidth * position), 0.0, vidWidth);
          mergeContext.drawImage(vid, colStart+vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
          requestAnimationFrame(drawLoop);


          var arrowLength = 0.09 * vidHeight;
          var arrowheadWidth = 0.025 * vidHeight;
          var arrowheadLength = 0.04 * vidHeight;
          var arrowPosY = vidHeight / 10;
          var arrowWidth = 0.007 * vidHeight;
          var currX = vidWidth * position;

          // Draw circle
          mergeContext.arc(currX, arrowPosY, arrowLength*0.7, 0, Math.PI * 2, false);
          mergeContext.fillStyle = "#FFD79340";
          mergeContext.fill()
          //mergeContext.strokeStyle = "#444444";
          //mergeContext.stroke()

          // Draw border
          mergeContext.beginPath();
          mergeContext.moveTo(vidWidth*position, 0);
          mergeContext.lineTo(vidWidth*position, vidHeight);
          mergeContext.closePath()
          mergeContext.strokeStyle = "#444444";
          mergeContext.lineWidth = 5;
          mergeContext.stroke();

          // Draw arrow
          mergeContext.beginPath();
          mergeContext.moveTo(currX, arrowPosY - arrowWidth/2);

          // Move right until meeting arrow head
          mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowWidth/2);

          // Draw right arrow head
          mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowheadWidth/2);
          mergeContext.lineTo(currX + arrowLength/2, arrowPosY);
          mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowheadWidth/2);
          mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowWidth/2);

          // Go back to the left until meeting left arrow head
          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowWidth/2);

          // Draw left arrow head
          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowheadWidth/2);
          mergeContext.lineTo(currX - arrowLength/2, arrowPosY);
          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY  - arrowheadWidth/2);
          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY);

          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY - arrowWidth/2);
          mergeContext.lineTo(currX, arrowPosY - arrowWidth/2);

          mergeContext.closePath();

          mergeContext.fillStyle = "#444444";
          mergeContext.fill();
      }
      requestAnimationFrame(drawLoop);
  }
}

// After the document loads, play the videos
function ondocumentready() {
  [...document.querySelectorAll('video.video-compare')].forEach(element => {
    // Add listener to onplay
    function loadeddata () {
      const canvas = document.createElement("canvas");
      element.parentNode.insertBefore(canvas, element.nextSibling);
      element.height = 0;
      element.style.position = "absolute";
      canvas.style.aspectRatio = `${element.videoWidth/2}/${element.videoHeight}`;
      canvas.width = element.videoWidth/2;
      canvas.height = element.videoHeight;
      canvas.classList.add("video-compare");
      element.play();
      playMergeVid(element, canvas);
    }
    if (element.readyState > 3) {
      loadeddata();
    } else {
      element.addEventListener("loadeddata", loadeddata);
    }
  });

  [...document.querySelectorAll('input[data-control-video]')].forEach(element => {
    const video = document.getElementById(element.getAttribute("data-control-video"));
    const sliderImagesRoot = document.getElementById(element.getAttribute("data-control-slider-images"));
    element.addEventListener("input", function() { 
      const value = Math.min(100, Math.max(0, element.value));
      const slice = video.duration * value / 100;
      video.currentTime = ""+slice;
      // Apply color to active borders
      if (sliderImagesRoot) {
        const el1 = Math.min(Math.floor(value / 100 * (sliderImagesRoot.children.length-1)), sliderImagesRoot.children.length-2);
        const offset = (sliderImagesRoot.children.length-1)*value/100 - el1;
        for (var i = 0; i < sliderImagesRoot.children.length; i++) {
          sliderImagesRoot.children[i].style.setProperty("--active-weight", "0%");
          if (i === el1) {
            sliderImagesRoot.children[i].style.setProperty("--active-weight", `${100*(1-offset)}%`);
          }
          if (i === el1+1) {
            sliderImagesRoot.children[i].style.setProperty("--active-weight", `${100*offset}%`);
          }
        }
      }
    });

  });
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', ondocumentready);
} else {
  ondocumentready();
}
