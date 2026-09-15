import React, { useState, useMemo, useEffect } from 'react';

export default function HtmlDashboard(componentProps) {
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  let scopeProps = {};
  try {
    if (typeof props !== 'undefined' && props) {
      scopeProps = props;
    }
  } catch (e) {}

  const p = {
    ...scopeProps,
    ...(componentProps || {}),
    ...((componentProps && componentProps.props) || {}),
  };

  const htmlContent = p.html_content;
  const title = p.title || '데이터 분석 대시보드';
  const height = p.height || '80vh';

  const blobUrl = useMemo(() => {
    if (!htmlContent) return null;
    const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' });
    return URL.createObjectURL(blob);
  }, [htmlContent]);

  useEffect(() => {
    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [blobUrl]);

  const containerStyle = {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    borderRadius: '12px',
    overflow: 'hidden',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
    border: '1px solid rgba(226, 232, 240, 0.8)',
    background: '#ffffff',
    margin: '8px 0',
    transition: 'all 0.3s ease',
  };

  const headerStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 16px',
    background: 'linear-gradient(135deg, #1E3A8A 0%, #2563EB 60%, #3B82F6 100%)',
    color: '#ffffff',
    fontSize: '13px',
    fontWeight: '600',
    letterSpacing: '-0.2px',
    fontFamily: "'Pretendard', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    flexShrink: 0,
    cursor: 'pointer',
    userSelect: 'none',
  };

  const titleWrapperStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    flex: 1,
  };

  const badgeStyle = {
    fontSize: '10px',
    fontWeight: '700',
    background: 'rgba(255, 255, 255, 0.2)',
    padding: '2px 7px',
    borderRadius: '12px',
    letterSpacing: '0.5px',
    textTransform: 'uppercase',
    flexShrink: 0,
  };

  const chevronStyle = {
    fontSize: '14px',
    transition: 'transform 0.3s ease',
    transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
    marginRight: '6px',
    flexShrink: 0,
  };

  const btnGroupStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    flexShrink: 0,
  };

  const btnStyle = {
    padding: '4px 12px',
    borderRadius: '6px',
    border: '1px solid rgba(255, 255, 255, 0.35)',
    background: 'rgba(255, 255, 255, 0.15)',
    color: '#ffffff',
    fontSize: '11px',
    fontWeight: '600',
    cursor: 'pointer',
    textDecoration: 'none',
    transition: 'all 0.2s ease',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    flexShrink: 0,
  };

  const bodyStyle = {
    position: 'relative',
    width: '100%',
    overflow: 'hidden',
    transition: 'max-height 0.4s ease, opacity 0.3s ease',
    maxHeight: collapsed ? '0px' : '700px',
    opacity: collapsed ? 0 : 1,
  };

  const iframeStyle = {
    width: '100%',
    height: '600px',
    border: 'none',
    display: 'block',
  };

  const loaderStyle = {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: loading && !collapsed ? 'flex' : 'none',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#ffffff',
    color: '#64748b',
    fontSize: '14px',
    zIndex: 1,
  };

  if (!blobUrl) {
    return (
      <div style={{ padding: '24px', color: '#ef4444', textAlign: 'center', background: '#fee2e2', borderRadius: '8px' }}>
        ⚠️ 대시보드 HTML 콘텐츠가 제공되지 않았습니다.
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <div
        style={headerStyle}
        onClick={() => setCollapsed(!collapsed)}
      >
        <div style={titleWrapperStyle}>
          <span style={chevronStyle}>{collapsed ? '▶' : '▼'}</span>
          <span style={badgeStyle}>Interactive</span>
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{title}</span>
        </div>
        <div style={btnGroupStyle}>
          <button
            style={btnStyle}
            onClick={(e) => { e.stopPropagation(); window.open(blobUrl, '_blank'); }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)')}
            title="새 탭에서 전체 화면으로 열기"
          >
            ↗ 새 탭
          </button>
          <button
            style={{...btnStyle, fontSize: '13px', padding: '4px 8px'}}
            onClick={(e) => { e.stopPropagation(); setCollapsed(!collapsed); }}
            title={collapsed ? '펼치기' : '접기'}
          >
            {collapsed ? '▼' : '▲'}
          </button>
        </div>
      </div>
      <div style={bodyStyle}>
        {loading && !collapsed && (
          <div style={loaderStyle}>
            ⏳ 대시보드를 불러오는 중입니다...
          </div>
        )}
        {!collapsed && (
          <iframe
            src={blobUrl}
            style={iframeStyle}
            title={title}
            onLoad={() => setLoading(false)}
            sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
          />
        )}
      </div>
    </div>
  );
}
