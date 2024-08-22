function whenDOMReady() {
    if (location.pathname == '/essay/'){
        setTimeout(
            () => { 
                console.log("DOMContentLoaded了！2222");
                btf.loadLightbox(document.querySelectorAll('#icat-bber img'));
                window.lazyLoadInstance && window.lazyLoadInstance.update();
                setTimeout(
                    () => { 
                        console.log("DOMContentLoaded了！3333");
                        reflashWaterFall();
                        changeTime();
                    }
                , 300)
            }
        , 300)
    }   
}
document.addEventListener("DOMContentLoaded", whenDOMReady),
document.addEventListener("pjax:complete", whenDOMReady),

// 适配pjax

window.onresize = () => {
    if (location.pathname == '/essay/'){
        console.log("窗口大小改变了！");
        waterfall('#waterfall');
    }
};

// 自适应

function timeDiff(timeObj, today){
    const timeObjUTC = Date.UTC(timeObj.getFullYear(), timeObj.getMonth(), timeObj.getDate());
    const todayUTC = Date.UTC(today.getFullYear(), today.getMonth(), today.getDate());

    const timeDiff = todayUTC - timeObjUTC;
    return Math.floor(timeDiff / (1000 * 3600 * 24));
}
function changeTime() {
    const timeElements = Array.from(document.getElementsByTagName("time"));
    const currentDate = new Date();

    timeElements.forEach(timeElement => {
    const datetime = timeElement.getAttribute("datetime");
    const timeObj = new Date(datetime);
    const daysDiff = timeDiff(timeObj, currentDate);

    let timeString;
    if (daysDiff === 0) {
        timeString = `最近`;
    } else if (daysDiff === 1) {
        timeString = `昨天`;
    } else if (daysDiff === 2) {
        timeString = `前天`;
    } else if (daysDiff <= 7) {
        timeString = `${daysDiff}天前`;
    } else {
        if (timeObj.getFullYear() !== currentDate.getFullYear()) {
        timeString = `${timeObj.getFullYear()}/${timeObj.getMonth() + 1}/${timeObj.getDate()}`;
        } else {
        timeString = `${timeObj.getMonth() + 1}/${timeObj.getDate()}`;
        }
    }
    timeElement.textContent = timeString;
    });
}
function reflashWaterFall() {
    document.querySelector("#waterfall") &&
    setTimeout(function() {
        waterfall("#waterfall");
        document.getElementById("waterfall")
        .classList.add("show");
    }, 500);
} // 加载显示 - 即刻短文
function commentText(txt) {
    const inputs = ["#wl-edit", ".el-textarea__inner"];
    for (let i = 0; i < inputs.length; i++) {
    let el = document.querySelector(inputs[i]);
    if (el != null) {
        el.dispatchEvent(new Event('input', { bubble: true, cancelable: true }));
        el.value = '> ' + txt.replace(/\n/g, '\n> ') + '\n\n';
        el.focus();
        el.setSelectionRange(-1, -1);
    }
    }
} // 引用评论跳转 - 即刻短文
