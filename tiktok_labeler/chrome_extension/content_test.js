/**
 * 測試版本 - 超簡單標註系統
 */

console.log('🚀 TSAR-RAPTOR 擴展已載入！');

// 立即顯示大大的提示
alert('🚀 TSAR-RAPTOR 標註系統已啟動！\n\n按鍵:\n← 左 = REAL\n→ 右 = AI\n↑ 上 = UNCERTAIN\n↓ 下 = MOVIE');

// 監聽鍵盤
document.addEventListener('keydown', function(e) {
    const url = window.location.href;

    if (!url.includes('tiktok.com')) return;

    let label = null;
    let emoji = '';

    if (e.key === 'ArrowLeft') {
        label = 'REAL';
        emoji = '✅';
    } else if (e.key === 'ArrowRight') {
        label = 'AI';
        emoji = '🤖';
    } else if (e.key === 'ArrowUp') {
        label = 'UNCERTAIN';
        emoji = '❓';
    } else if (e.key === 'ArrowDown') {
        label = 'EXCLUDE';
        emoji = '🎬';
    }

    if (label) {
        // 顯示大大的提示
        showBigEmoji(emoji, label);

        // 發送到後端
        sendLabel(url, label);

        console.log(`✅ 標註: ${label} - ${url}`);
    }
});

// 顯示大大的 emoji
function showBigEmoji(emoji, text) {
    const div = document.createElement('div');
    div.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 200px;
        z-index: 999999;
        background: rgba(0,0,0,0.8);
        padding: 50px;
        border-radius: 20px;
        color: white;
        text-align: center;
    `;
    div.innerHTML = `${emoji}<br><div style="font-size: 40px; margin-top: 20px;">${text}</div>`;
    document.body.appendChild(div);

    setTimeout(() => div.remove(), 1500);
}

// 發送標註到後端
function sendLabel(url, label) {
    fetch('http://127.0.0.1:5000/api/label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            url: url,
            label: label,
            timestamp: new Date().toISOString()
        })
    })
    .then(res => res.json())
    .then(data => console.log('✅ 後端回應:', data))
    .catch(err => console.error('❌ 後端錯誤:', err));
}

console.log('👂 鍵盤監聽已啟動');
