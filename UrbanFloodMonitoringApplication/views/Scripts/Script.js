  
        // ====== Khởi tạo bản đồ ======
        const map = L.map('map', {
            zoomControl: true,
            preferCanvas: true
        });
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
        }).addTo(map);
        map.setView([16.0471, 108.2062], 6);
        let currentMarker = null;
        let pathLine = L.polyline([], {
            color: '#3b82f6',
            weight: 3,
            opacity: .8
        }).addTo(map);
        // Nhóm layer để quản lý các vòng tròn ngập
        const floodLayerGroup = L.layerGroup().addTo(map);
        // Theo dõi tất cả vòng tròn đã thêm để lọc
        const floodCircles = [];
        // { circle, levelSet, maxLevel, baseColor, baseFill }
        // ========== Bộ lọc theo chú thích ==========
        // filterMode: 'none' | '0' | '1-2' | '3+'
        let filterMode = 'none';

        function applyFilter() {
            floodCircles.forEach(obj => {
                const {
                    circle,
                    maxLevel,
                    baseColor,
                    baseFill
                } = obj;
                // Quy tắc mapping vào nhóm
                const bucket = maxLevel === 0 ? '0' : (maxLevel <= 2 ? '1-2' : '3+');
                if (filterMode === 'none') {
                    // khôi phục màu
                    circle.setStyle({
                        color: baseColor,
                        fillColor: baseFill,
                        opacity: 1,
                        fillOpacity: .4
                    });
                } else if (filterMode === bucket) {
                    circle.setStyle({
                        color: baseColor,
                        fillColor: baseFill,
                        opacity: 1,
                        fillOpacity: .5
                    });
                    circle.bringToFront();
                } else {
                    // xám mờ
                    circle.setStyle({
                        color: '#6b7280',
                        fillColor: '#6b7280',
                        opacity: .5,
                        fillOpacity: .15
                    });
                }
            });
        }

        function setFilter(mode) {
            filterMode = mode;
            document.querySelectorAll('.legend-item').forEach(el => el.classList.toggle('active', el.dataset.range === mode));
            applyFilter();
        }

        // Click nền để bỏ lọc
        map.on('click', () => setFilter('none'));

        // Xóa lọc qua nút
        document.getElementById('btnClearFilter').addEventListener('click', () => setFilter('none'));

        // Bind click cho legend
        document.querySelectorAll('.legend-item').forEach(el => {
            el.addEventListener('click', (e) => {
                e.stopPropagation(); // tránh bị map click clear luôn
                const key = el.dataset.range;
                setFilter(filterMode === key ? 'none' : key);
            });
        });

        // ====== Hiệu ứng pulse nhỏ khi thêm điểm ======
        function addPulse(latLng) {
            const tpl = document.getElementById('pulse-template');
            const node = tpl.content.firstElementChild.cloneNode(true);
            const pane = map.getPanes().overlayPane;
            pane.appendChild(node);

            function updatePos() {
                const p = map.latLngToLayerPoint(latLng);
                node.style.transform = `translate(${p.x}px, ${p.y}px)`;
            }
            updatePos();
            const onZoom = () => requestAnimationFrame(updatePos);
            map.on('zoom viewreset move', onZoom);
            setTimeout(() => {
                map.off('zoom viewreset move', onZoom);
                node.remove();
            }, 1100);
        }

        // ====== Hàm showPoint đã nâng cấp ======
        function showPoint(lat, lng, timestamp, prediction) {
            if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
            // prediction có thể là mảng (list các mức). Ở code cũ bạn lấy prediction[0], tôi giữ nguyên để tương thích:
            const pred = prediction;
            // Chuẩn hóa thành mảng số nguyên không âm
            const levels = (Array.isArray(pred) ? pred : [pred])
                .map(n => Number(n))
                .filter(n => Number.isInteger(n) && n >= 0);
            if (!levels.length) return;

            // Sắp xếp giảm dần và loại trùng
            const uniqueDesc = [...new Set(levels)].sort((a, b) => b - a);
            const maxLevel = uniqueDesc[0];

            // Tạo text hiển thị theo yêu cầu:
            // - Nếu 1 mức: “đang ngập ở mức X”
            // - Nếu nhiều mức: “có thể đang ngập tới mức max” + liệt kê mức khác tăng dần
            let text;
            if (uniqueDesc.length === 1) {
                text = `Thời gian: ${timestamp}<br>Dự đoán khu vực đang ngập ở mức ${maxLevel}`;
            } else {
                const others = uniqueDesc.slice(1).sort((a, b) => a - b);
                const othersText = others.map(l => `mức ${l}`).join(', ');
                text = `Thời gian: ${timestamp}<br>` +
                    `Dự đoán khu vực có thể đang ngập tới mức ${maxLevel} ` +
                    `có vị trí ngập ${othersText}`;
            }

            // Màu: mức 0 xanh; tồn tại mức >2 → đỏ; nếu tối đa chỉ 2 → vàng
            const hasRed = uniqueDesc.some(l => l > 2);
            const baseColor = (maxLevel === 0) ? 'green' : (hasRed ? 'red' : 'orange');
            const baseFill = (maxLevel === 0) ? '#22c55e' : (hasRed ? '#ef4444' : '#ffcc00');
            const latLng = [lat, lng];

            // Vẽ vòng tròn
            const circle = L.circle(latLng, {
                color: baseColor,
                fillColor: baseFill,
                fillOpacity: 0.4,
                radius: 18,
                weight: 2
            }).bindPopup(text);
            circle.addTo(floodLayerGroup);

            // Lưu để phục vụ lọc
            floodCircles.push({
                circle,
                levelSet: new Set(uniqueDesc),
                maxLevel,
                baseColor,
                baseFill
            });

            // Áp bộ lọc hiện tại (nếu đang lọc)
            applyFilter();

            // Popup mở nhẹ khi vừa thêm
            circle.on('add', () => {
                addPulse(latLng);
                // Không auto-open popup để tránh spam; bấm vào từng điểm để xem.
            });

            // Cập nhật marker chính
            if (!window.currentMarker) {
                window.currentMarker = L.marker(latLng).addTo(map);
            } else {
                window.currentMarker.setLatLng(latLng);
            }

            // Cập nhật đường đi
            if (window.pathLine) {
                window.pathLine.addLatLng(latLng);
            }

            // Giữ zoom tối thiểu 17
            const currentZoom = map.getZoom();
            map.setView(latLng, Math.max(currentZoom, 17));
        }
        // ====== Helpers chung ======
    // Khóa/mở khóa mọi tương tác trang
    function setAppBlocking(on, title = '', sub = '') {
        const overlay = document.getElementById('appOverlay');
        const t = document.getElementById('overlayTitle');
        const s = document.getElementById('overlaySub');
        // Disable/enable tất cả nút và input
        const controls = document.querySelectorAll('button, input, select, textarea');
        controls.forEach(el => {
            el.disabled = !!on;
        });
        if (on) {
            t.textContent = title || 'Đang xử lý…';
            s.textContent = sub || 'Vui lòng đợi trong giây lát.';
            overlay.classList.add('show');
            overlay.setAttribute('aria-hidden', 'false');
        } else {
            overlay.classList.remove('show');
            overlay.setAttribute('aria-hidden', 'true');
        }
    }
        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const result = reader.result;
                    const idx = result.indexOf(',');
                    resolve(idx >= 0 ? result.slice(idx + 1) : result);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        }

        function isSecureContextForGeo() {
            return window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost';
        }

        function formatTimestamp(d = new Date()) {
            const y = d.getFullYear();
            const m = String(d.getMonth() + 1).padStart(2, '0');
            const da = String(d.getDate()).padStart(2, '0');
            const h = String(d.getHours()).padStart(2, '0');
            const mi = String(d.getMinutes()).padStart(2, '0');
            const s = String(d.getSeconds()).padStart(2, '0');
            return `${y}-${m}-${da} ${h}:${mi}:${s}`;
        }

        // ====== Gửi ảnh ======
        
        const btnSendImage = document.getElementById('btnSendImage');
        const imageInput = document.getElementById('imageInput');
        const imgStatus = document.getElementById('imgStatus');
        btnSendImage.addEventListener('click', async () => {
            imgStatus.textContent = '';
            imgStatus.className = 'status';
        try {
            const file = imageInput.files && imageInput.files[0];
            if (!file) {
                imgStatus.textContent = 'Vui lòng chọn ảnh trước.';
                imgStatus.classList.add('err');
                return;
            }
            // Khóa UI + hiển thị overlay đang dự đoán
            setAppBlocking(true, 'AI đang dự đoán mức ngập…', 'Hệ thống đang phân tích ảnh của bạn.');
            const base64 = await fileToBase64(file);
            const res = await fetch('/api/send-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_base64: base64,
                    sent_at: Date.now()
                })
            });
            if (!res.ok) throw new Error('Gửi ảnh thất bại');
            const data = await res.json();
            // Cập nhật thông báo theo yêu cầu
            if (data.ok) {
                // Giữ overlay ~800ms cho cảm giác “xong tác vụ”
                document.getElementById('overlayTitle').textContent = 'Đã dự đoán xong mức ngập tại khu vực';
                document.getElementById('overlaySub').textContent = 'Nhấn “Gửi vị trí” để ghi nhận.';
                setTimeout(() => setAppBlocking(false), 800);
                imgStatus.textContent = 'Đã dự đoán xong mức ngập tại khu vực, nhấn Gửi vị trí để ghi nhận.';
                imgStatus.classList.add('ok');
            } else {
                throw new Error(data.error || 'Gửi ảnh thất bại');
            }
        } catch (e) {
            setAppBlocking(false);
            imgStatus.textContent = 'Lỗi: ' + e.message;
            imgStatus.classList.add('err');
        }
    });

        // ====== Gửi vị trí một lần ======
    const btnSendLocation = document.getElementById('btnSendLocation');
    const locStatus = document.getElementById('locStatus');
    btnSendLocation.addEventListener('click', async () => {
        locStatus.textContent = '';
        locStatus.className = 'status';
        if (!isSecureContextForGeo()) {
            locStatus.textContent = 'Cần mở bằng HTTPS hoặc localhost để lấy vị trí.';
            locStatus.classList.add('err');
            return;
        }
        if (!('geolocation' in navigator)) {
            locStatus.textContent = 'Trình duyệt không hỗ trợ Geolocation.';
            locStatus.classList.add('err');
            return;
        }
        try {
            // Khóa UI + hiển thị overlay "đang ghi nhận"
            setAppBlocking(true, 'Đang ghi nhận kết quả…', 'Đang gửi vị trí và lưu vào hệ thống.');
            const pos = await new Promise((resolve, reject) => navigator.geolocation.getCurrentPosition(resolve, reject, {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }));
            const ts = formatTimestamp(new Date());
            const payload = {
                lat: pos.coords.latitude,
                lng: pos.coords.longitude,
                timestamp: ts
            };
            const res = await fetch('/api/send-location', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error('Gửi vị trí thất bại');
            const data = await res.json();
            if (data.ok) {
                // Đợi 600–800ms rồi đóng overlay để UX mượt
                document.getElementById('overlayTitle').textContent = 'Đã ghi nhận kết quả';
                document.getElementById('overlaySub').textContent = 'Cảm ơn bạn đã đóng góp dữ liệu.';
                setTimeout(() => setAppBlocking(false), 700);
                locStatus.textContent = `Đã gửi vị trí: (${payload.lat.toFixed(6)}, ${payload.lng.toFixed(6)})`;
                locStatus.classList.add('ok');
            } else {
                throw new Error(data.error || 'Gửi vị trí thất bại');
            }
        } catch (e) {
            setAppBlocking(false);
            locStatus.textContent = 'Lỗi: ' + e.message;
            locStatus.classList.add('err');
        }
    });
        // ====== SSE nhận dữ liệu và vẽ lên map ======
        const mapStatus = document.getElementById('mapStatus');
        try {
            const sse = new EventSource('/events');
            sse.onmessage = (ev) => {
                try {
                    const payload = JSON.parse(ev.data);
                    if (payload && Number.isFinite(payload.lat) && Number.isFinite(payload.lng)) {
                        showPoint(payload.lat, payload.lng, payload.timestamp, payload.prediction);
                        mapStatus.textContent = `Đã cập nhật vị trí: (${payload.lat.toFixed(6)}, ${payload.lng.toFixed(6)})`;
                        mapStatus.className = 'status ok';
                    }
                } catch (e) {
                    console.error('Bad SSE JSON:', e, ev.data);
                }
            };
            sse.onerror = () => {
                mapStatus.textContent = 'Mất kết nối realtime, đang thử lại...';
                mapStatus.className = 'status err';
            };
        } catch (e) {
            mapStatus.textContent = 'Không thể mở kênh realtime.';
            mapStatus.className = 'status err';
        }

        // ====== Broadcast dữ liệu đã lưu ======
        const btnBroadcastStored = document.getElementById('btnBroadcastStored');
        const storedInfo = document.getElementById('storedInfo');

        async function refreshStoredCount() {
            try {
                const res = await fetch('/api/stored-count');
                const data = await res.json();
                if (data.ok) {
                    storedInfo.textContent = `Hiện có ${data.count} bản ghi trong bộ nhớ.`;
                    storedInfo.className = 'status';
                }
            } catch {}
        }

        refreshStoredCount();
        btnBroadcastStored.addEventListener('click', async () => {
            btnBroadcastStored.disabled = true;
            storedInfo.textContent = 'Đang phát dữ liệu đã lưu...';
            storedInfo.className = 'status';
            try {
                const res = await fetch('/api/broadcast-stored', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        delayMs: 300
                    })
                });
                const data = await res.json();
                if (data.ok) {
                    storedInfo.textContent = `Đã phát ${data.broadcasted} bản ghi lên bản đồ.`;
                    storedInfo.className = 'status ok';
                } else {
                    storedInfo.textContent = 'Không phát được dữ liệu.';
                    storedInfo.className = 'status err';
                }
            } catch (e) {
                storedInfo.textContent = 'Lỗi khi phát dữ liệu: ' + e.message;
                storedInfo.className = 'status err';
            } finally {
                btnBroadcastStored.disabled = false;
                refreshStoredCount();
            }
        });

        // ====== Theo dõi vị trí liên tục ======
        let watchId = null;
        let isTracking = false;
        let myLocationMarker = null;
        let myLocationCircle = null;
        const btnToggleTracking = document.getElementById('btnToggleTracking');
        const trackStatus = document.getElementById('trackStatus');
        const chkAutoSend = document.getElementById('chkAutoSend');
        let lastSentTime = 0;
        const SEND_INTERVAL = 5000;

        function updateMyLocation(lat, lng, accuracy) {
            const latLng = [lat, lng];
            // marker xanh “vị trí của tôi”
            if (!myLocationMarker) {
                const myIcon = L.divIcon({
                    className: 'my-location-icon',
                    html: '<div style="background-color:#3b82f6;width:16px;height:16px;border-radius:50%;border:3px solid white;box-shadow:0 0 8px rgba(59,130,246,.7);"></div>',
                    iconSize: [16, 16],
                    iconAnchor: [8, 8]
                });
                myLocationMarker = L.marker(latLng, {
                    icon: myIcon
                }).addTo(map);
                myLocationMarker.bindPopup('Vị trí của bạn');
            } else {
                myLocationMarker.setLatLng(latLng);
            }

            // vòng accuracy
            if (!myLocationCircle) {
                myLocationCircle = L.circle(latLng, {
                    color: '#3b82f6',
                    fillColor: '#3b82f6',
                    fillOpacity: 0.12,
                    radius: accuracy
                }).addTo(map);
            } else {
                myLocationCircle.setLatLng(latLng);
                myLocationCircle.setRadius(accuracy);
            }

            // pan hợp lý
            const currentZoom = map.getZoom();
            if (currentZoom < 15) map.setView(latLng, 15);
            else map.panTo(latLng);
        }

        async function sendLocationToServer(lat, lng) {
            try {
                const payload = {
                    lat,
                    lng,
                    timestamp: formatTimestamp()
                };
                const res = await fetch('/api/send-location', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    console.warn('Send location failed');
                }
            } catch (e) {
                console.error('Lỗi gửi vị trí tự động:', e);
            }
        }

        function startTracking() {
            if (!isSecureContextForGeo()) {
                trackStatus.textContent = 'Trình duyệt yêu cầu HTTPS hoặc localhost để lấy vị trí.';
                trackStatus.className = 'status err';
                return;
            }
            if (!('geolocation' in navigator)) {
                trackStatus.textContent = 'Trình duyệt không hỗ trợ Geolocation.';
                trackStatus.className = 'status err';
                return;
            }
            watchId = navigator.geolocation.watchPosition(async (pos) => {
                const lat = pos.coords.latitude;
                const lng = pos.coords.longitude;
                const acc = pos.coords.accuracy;
                updateMyLocation(lat, lng, acc);
                trackStatus.textContent = `Vị trí: (${lat.toFixed(6)}, ${lng.toFixed(6)}) - Độ chính xác: ${acc.toFixed(0)}m`;
                trackStatus.className = 'status ok';
                if (chkAutoSend.checked) {
                    const now = Date.now();
                    if (now - lastSentTime >= SEND_INTERVAL) {
                        lastSentTime = now;
                        await sendLocationToServer(lat, lng);
                    }
                }
            }, (error) => {
                trackStatus.textContent = `Lỗi: ${error.message}`;
                trackStatus.className = 'status err';
            }, {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            });
            isTracking = true;
            btnToggleTracking.textContent = 'Tắt theo dõi vị trí';
            btnToggleTracking.classList.remove('warn');
            btnToggleTracking.classList.add('danger');
        }

        function stopTracking() {
            if (watchId !== null) {
                navigator.geolocation.clearWatch(watchId);
                watchId = null;
            }
            isTracking = false;
            btnToggleTracking.textContent = 'Bật theo dõi vị trí';
            btnToggleTracking.classList.remove('danger');
            btnToggleTracking.classList.add('warn');
            trackStatus.textContent = 'Đã tắt theo dõi vị trí.';
            trackStatus.className = 'status';
        }

        btnToggleTracking?.addEventListener('click', () => {
            if (isTracking) stopTracking();
            else startTracking();
        });

        window.addEventListener('beforeunload', () => {
            if (isTracking) stopTracking();
        });