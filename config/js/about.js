function whenDOMReady() {
	if (location.pathname.startsWith('/about/')) meuicat.Introduction(), meuicat.runtimen();
}
let lastSayHello = "";
var meuicat = {
	Introduction: function() {
		const e = [
				"🤖️ 数码科技爱好者",
				"🔍 分享与热心帮助",
				"🏠 智能家居小能手",
				"🔨 设计开发一条龙",
				"📷 人文摄影的坚定者",
				"🏃 脚踏实地行动派",
				"📚 热爱阅读的书虫迷",
				"🎵 薛之谦热爱fans",
				"🏋️‍♀️ 坚韧不拔的健身达人",
				"🍜 走哪吃哪的美食迷",
				// "🎮 Minecraft骨灰级玩家",
				// "👨‍🍳 一位爱做饭的程序猿",
			],
			t = document.getElementById("Introduction");
		let o = e[Math.floor(Math.random() * e.length)];
		for (; o === lastSayHello;) o = e[Math.floor(Math.random() * e.length)];
		(t.textContent = o), (lastSayHello = o);
	}, // about 个人介绍词
	runtimen: function() {
		let t = new Date("2021/10/15 00:00:00")
			.getTime(),
			n = new Date()
			.getTime(),
			a = Math.round((n - t) / 1e3),
			l = (a / 7884e4)
			.toFixed(2);
		let c = document.getElementById("run-time");
		c && (c.innerHTML = `已稳定运行 ${l} 坤年 🏀`),
			setTimeout(meuicat.runtime, 1e3);
	}, // about 运行时间
}

whenDOMReady() // 打开网站先执行一次
document.addEventListener("pjax:complete", whenDOMReady) // pjax加载完成（切换页面）后再执行一次



