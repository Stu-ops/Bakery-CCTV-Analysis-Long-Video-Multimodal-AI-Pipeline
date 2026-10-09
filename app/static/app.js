(function() {
    let ws = null;
    let wsEventCount = 0;
    let eventTimestamps = [];
    let firstEvent = false;
    let entryBuckets = new Array(30).fill(0);

    const $badge = document.getElementById('ws-badge');
    const $label = document.getElementById('ws-label');
    const $log = document.getElementById('event-log');

    // XSS Remediation: Secure HTML encoder
    function sanitizeHTML(str) {
        if (str === null || str === undefined) return '';
        return String(str).replace(/[&<>"'`=\/]/g, function (s) {
            switch (s) {
                case '&': return '&amp;';
                case '<': return '&lt;';
                case '>': return '&gt;';
                case '"': return '&quot;';
                case "'": return '&#39;';
                case '/': return '&#x2F;';
                case '`': return '&#x60;';
                case '=': return '&#x3D;';
                default: return s;
            }
        });
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/ws/live`);

        ws.onopen = () => {
            $badge.className = 'status-badge connected';
            setText('ws-label', 'Connected');
            setText('ps-status', 'Live Broadcast');
        };
        ws.onclose = () => {
            $badge.className = 'status-badge';
            setText('ws-label', 'Reconnecting…');
            setText('ps-status', 'Reconnecting');
            
            // Clear active states on disconnect
            document.querySelectorAll('.cam-card').forEach(c => c.classList.remove('active-feed'));
            
            setTimeout(connect, 2000);
        };
        ws.onerror = () => {
            $badge.className = 'status-badge';
            setText('ws-label', 'Error');
        };
        ws.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);
                handleMsg(msg);
            } catch(err) {
                console.warn('WS message parse exception:', err);
            }
        };
    }

    function handleMsg(msg) {
        if (msg.type === 'event') handleEvent(msg);
        else if (msg.type === 'metrics') handleMetrics(msg.data);
        else if (msg.type === 'snapshot') handleSnapshot(msg);
        else if (msg.type === 'heartbeat') handleHeartbeat(msg);
        else if (msg.type === 'video_frame') handleVideoFrame(msg);
    }

    function handleVideoFrame(msg) {
        const img = document.getElementById('feed-' + sanitizeHTML(msg.camera_id));
        const card = document.getElementById('camcard-' + sanitizeHTML(msg.camera_id));
        if (img && card) {
            img.style.display = 'block';
            if (img.nextElementSibling) img.nextElementSibling.style.display = 'none';
            img.src = 'data:image/jpeg;base64,' + msg.frame; 
            
            // Activate glowing badge
            card.classList.add('active-feed');
        }
    }

    function handleSnapshot(msg) {
        wsEventCount = msg.event_count || 0;
        setText('f-ws', wsEventCount);
        if (msg.recent_events) {
            msg.recent_events.slice(-20).forEach(ev => {
                if (ev.data) addToLog(ev.data, ev.server_time);
            });
        }
        fetchMetrics();
        fetchHeatmap();
    }

    function handleEvent(msg) {
        wsEventCount = msg.seq || wsEventCount + 1;
        setText('h-evt-count', wsEventCount);
        setText('f-ws', wsEventCount);

        const now = Date.now();
        eventTimestamps.push(now);
        eventTimestamps = eventTimestamps.filter(t => now - t < 10000);
        const rate = (eventTimestamps.length / 10).toFixed(1);
        setText('event-rate', `${rate} msg/s`);
        setText('ps-rate', rate);
        setText('ps-status', 'Processing Pipeline');
        setText('ps-last', new Date().toLocaleTimeString());

        const evt = msg.data;
        if (evt && evt.event_type === 'ENTRY') {
            entryBuckets[29]++;
            renderSparkline();
        }

        addToLog(evt, msg.server_time);
        updateFooter(msg.server_time, msg.seq);
    }

    function handleHeartbeat(msg) {
        setText('ps-server', msg.server_time ? sanitizeHTML(new Date(msg.server_time).toLocaleTimeString()) : '--');
        setText('ps-clients', sanitizeHTML(msg.subscribers || 0));
    }

    function handleMetrics(data) {
        if (!data) return;

        setCard('v-visitors', data.unique_visitors, 'c-visitors');
        setCard('v-staff', data.staff_count, 'c-staff');

        if (data.conversion_rate !== undefined) {
            const pct = sanitizeHTML((data.conversion_rate * 100).toFixed(1)) + '%';
            const el = document.getElementById('v-conv');
            if (el && el.textContent !== pct) { 
                el.textContent = pct; 
                flashCard('c-conv'); 
            }
            setText('s-conv', sanitizeHTML(data.total_conversions || 0));
        }

        if (data.queue_stats) {
            const depth = data.queue_stats.current_depth || 0;
            const el = document.getElementById('v-queue');
            const card = document.getElementById('c-queue');
            
            if (el && el.textContent !== String(depth)) { 
                el.textContent = sanitizeHTML(depth); 
                flashCard('c-queue'); 
            }
            
            if (card) {
                if (depth >= 5) card.classList.add('warning-depth');
                else card.classList.remove('warning-depth');
            }
            setText('s-queue', sanitizeHTML((data.queue_stats.average_depth || 0.0).toFixed(1)));
        }

        if (data.total_entries !== undefined) {
            setText('f-db', sanitizeHTML((data.total_entries || 0) + (data.total_exits || 0)));
        }

        if (data.funnel && data.funnel.length > 0) {
            renderFunnel(data.funnel);
            setText('funnel-ts', 'Telemetry Sync: ' + sanitizeHTML(new Date().toLocaleTimeString()));
        }

        if (data.anomalies !== undefined) {
            renderAnomalies(data.anomalies);
        }
    }

    function setCard(valId, value, cardId) {
        const el = document.getElementById(valId);
        if (!el) return;
        const v = value !== undefined ? value : 0;
        if (el.textContent !== String(v)) {
            el.textContent = sanitizeHTML(v);
            flashCard(cardId);
        }
    }

    function flashCard(cardId) {
        const card = document.getElementById(cardId);
        if (card) {
            card.classList.add('flash-active');
            setTimeout(() => card.classList.remove('flash-active'), 800);
        }
    }

    function addToLog(event, serverTime) {
        if (!event) return;
        if (!firstEvent) { 
            if ($log) $log.textContent = ''; 
            firstEvent = true; 
        }

        const time = serverTime ? new Date(serverTime).toLocaleTimeString() : new Date().toLocaleTimeString();
        const type = event.event_type || 'UNKNOWN';
        const vid = event.visitor_id ? event.visitor_id.substring(0, 14) : '--';
        const zone = event.zone_id || event.zone || '';
        const conf = event.confidence ? event.confidence.toFixed(2) : '';

        const div = document.createElement('div');
        div.className = 'ev-item';
        
        let desc = `Visitor ${vid}`;
        if (type === 'ENTRY') desc = `Visitor ${vid} entered store`;
        else if (type === 'EXIT') desc = `Visitor ${vid} exited store`;
        else if (type === 'ZONE_ENTER') desc = `Visitor ${vid} entered ${zone}`;
        else if (type === 'ZONE_EXIT') desc = `Visitor ${vid} exited ${zone}`;
        else if (type === 'ZONE_DWELL') desc = `Visitor ${vid} dwelled in ${zone}`;
        else if (type === 'BILLING_QUEUE_JOIN') desc = `Visitor ${vid} joined billing queue`;
        else if (type === 'BILLING_QUEUE_EXIT') desc = `Visitor ${vid} exited billing queue`;
        else if (type === 'REENTRY') desc = `Visitor ${vid} returning visitor`;
        else if (type === 'STAFF_CLASSIFIED') desc = `Identity ${vid} classified as Staff`;
        else if (type === 'ROLE_CHANGED') {
            const role = event.new_role || 'STAFF';
            desc = `Identity ${vid} is now ${role}`;
        }

        const leftDiv = document.createElement('div');
        leftDiv.className = 'ev-left';
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'ev-time';
        timeSpan.textContent = time;
        
        const badgeSpan = document.createElement('span');
        badgeSpan.className = `ev-badge ${type}`;
        badgeSpan.textContent = type;
        
        const descSpan = document.createElement('span');
        descSpan.className = 'ev-desc';
        descSpan.textContent = desc;
        
        leftDiv.appendChild(timeSpan);
        leftDiv.appendChild(badgeSpan);
        leftDiv.appendChild(descSpan);
        
        div.appendChild(leftDiv);
        
        if (conf) {
            const confSpan = document.createElement('span');
            confSpan.className = 'ev-meta-text';
            confSpan.textContent = `c:${conf}`;
            div.appendChild(confSpan);
        }

        if ($log) {
            $log.prepend(div);
            while ($log.children.length > 50) $log.removeChild($log.lastChild);
        }
    }


    function renderFunnel(stages) {
        const container = document.getElementById('funnel-container');
        if (!container) return;
        
        container.textContent = ''; // Safely clear container
        
        const funnelDiv = document.createElement('div');
        funnelDiv.className = 'funnel-container';
        
        stages.forEach(s => {
            if (!s) return;
            const stage = s.stage || 'Unknown';
            const pct = s.percentage !== undefined ? s.percentage : 0;
            const count = s.count || 0;
            
            const stageDiv = document.createElement('div');
            stageDiv.className = 'funnel-stage';
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'funnel-label';
            labelDiv.textContent = stage;
            
            const barBg = document.createElement('div');
            barBg.className = 'funnel-bar-bg';
            
            const barFill = document.createElement('div');
            barFill.className = 'funnel-bar-fill';
            barFill.style.width = `${pct}%`;
            
            barBg.appendChild(barFill);
            
            const pctDiv = document.createElement('div');
            pctDiv.className = 'funnel-percentage';
            pctDiv.textContent = `${pct}% (${count})`;
            
            stageDiv.appendChild(labelDiv);
            stageDiv.appendChild(barBg);
            stageDiv.appendChild(pctDiv);
            
            funnelDiv.appendChild(stageDiv);
        });
        
        container.appendChild(funnelDiv);
    }

    function renderHeatmap(data) {
        if (!data || !data.zones || data.zones.length === 0) return;
        const container = document.getElementById('heatmap-container');
        if (!container) return;
        
        const confText = `Confidence: ${(data.data_confidence || 'UNKNOWN').toUpperCase()} (${data.total_sessions} Sessions)`;
        setText('heatmap-confidence', confText);

        container.textContent = '';
        
        const grid = document.createElement('div');
        grid.className = 'heatmap-grid';
        
        data.zones.forEach(z => {
            const hue = Math.round(210 - (z.normalized_score / 100) * 200); // 210 (Blue) to 10 (Red)
            
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            cell.style.borderBottomColor = `hsl(${hue}, 60%, 50%)`;
            
            const nameDiv = document.createElement('div');
            nameDiv.className = 'hm-name';
            nameDiv.textContent = z.zone_id;
            
            const scoreDiv = document.createElement('div');
            scoreDiv.className = 'hm-score';
            scoreDiv.style.color = `hsl(${hue}, 70%, 45%)`;
            scoreDiv.textContent = z.normalized_score.toFixed(0);
            
            const detailDiv = document.createElement('div');
            detailDiv.className = 'hm-detail';
            detailDiv.textContent = `${z.visit_count} visits · ${z.avg_dwell_seconds.toFixed(0)}s`;
            
            cell.appendChild(nameDiv);
            cell.appendChild(scoreDiv);
            cell.appendChild(detailDiv);
            
            grid.appendChild(cell);
        });
        
        container.appendChild(grid);
    }

    function renderAnomalies(anomalies) {
        const container = document.getElementById('anomaly-container');
        const badge = document.getElementById('anomaly-count');
        if (!container || !badge) return;
        
        container.textContent = '';
        
        if (!anomalies || anomalies.length === 0) {
            const emptyDiv = document.createElement('div');
            emptyDiv.className = 'empty-placeholder';
            emptyDiv.style.color = 'var(--color-success)';
            emptyDiv.style.height = '100px';
            
            emptyDiv.innerHTML = '<svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>';
            
            const span = document.createElement('span');
            span.textContent = 'Core Systems Healthy - No Anomalies Detected';
            emptyDiv.appendChild(span);
            
            container.appendChild(emptyDiv);
            
            badge.textContent = '0';
            badge.style.background = '#f3f4f6';
            badge.style.color = 'var(--text-muted)';
            return;
        }
        
        badge.textContent = anomalies.length;
        badge.style.background = '#fee2e2';
        badge.style.color = 'var(--color-danger)';

        anomalies.forEach(a => {
            if (!a) return;
            const sev = a.severity || 'INFO';
            const type = a.type || 'SYSTEM';
            const msg = typeof a === 'string' ? a : (a.message || 'Unknown anomaly');
            const action = a.suggested_action || '';
            
            const card = document.createElement('div');
            card.className = `anomaly-card ${sev}`;
            
            const headerRow = document.createElement('div');
            headerRow.className = 'anomaly-header-row';
            
            const typeBadge = document.createElement('span');
            typeBadge.className = `anomaly-type-badge ${sev}`;
            typeBadge.textContent = type;
            
            const sevBadge = document.createElement('span');
            sevBadge.className = `anomaly-type-badge ${sev}`;
            sevBadge.textContent = sev;
            
            headerRow.appendChild(typeBadge);
            headerRow.appendChild(sevBadge);
            
            const textDiv = document.createElement('div');
            textDiv.className = 'anomaly-text';
            textDiv.textContent = msg;
            
            card.appendChild(headerRow);
            card.appendChild(textDiv);
            
            if (action) {
                const actionDiv = document.createElement('div');
                actionDiv.className = 'anomaly-action-text';
                actionDiv.textContent = `↳ ${action}`;
                card.appendChild(actionDiv);
            }
            
            container.appendChild(card);
        });
    }

    function renderSparkline() {
        const max = Math.max(...entryBuckets, 1);
        const container = document.getElementById('sparkline');
        if (!container) return;
        
        container.textContent = '';
        
        entryBuckets.forEach((v, i) => {
            const h = Math.max(3, (v / max) * 45);
            const isLatest = (i === entryBuckets.length - 1);
            
            const bar = document.createElement('div');
            bar.className = `spark-bar ${isLatest ? 'latest-tick' : ''}`;
            bar.style.height = `${h}px`;
            
            container.appendChild(bar);
        });
    }

    // Rotate entry trend buckets every 4 seconds
    setInterval(() => {
        entryBuckets.push(0);
        if (entryBuckets.length > 30) entryBuckets.shift();
        renderSparkline();
    }, 4000);

    async function fetchMetrics() {
        try {
            const r = await fetch('/metrics');
            const d = await r.json();
            handleMetrics(d);
        } catch(e) { 
            console.warn('Fallback metrics fetch failed:', e); 
        }
    }

    async function fetchHeatmap() {
        try {
            const r = await fetch('/stores/STORE_BLR_002/heatmap');
            const d = await r.json();
            renderHeatmap(d);
        } catch(e) { 
            console.warn('Fallback heatmap fetch failed:', e); 
        }
    }

    function updateFooter(serverTime, count) {
        setText('f-ws', count || wsEventCount);
        setText('f-updated', sanitizeHTML(new Date().toLocaleTimeString()));
        if (serverTime) {
            setText('ps-server', sanitizeHTML(new Date(serverTime).toLocaleTimeString()));
        }
    }

    // Init
    connect();
    renderSparkline();
    // Regular REST sync backup every 10s
    setInterval(() => { 
        fetchMetrics(); 
        fetchHeatmap(); 
    }, 10000);
})();
