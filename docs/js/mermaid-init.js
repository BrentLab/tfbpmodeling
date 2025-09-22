document.addEventListener('DOMContentLoaded', function() {
    mermaid.initialize({
        startOnLoad: true,
        theme: 'default',
        themeVariables: {
            primaryColor: '#ff6b6b',
            primaryTextColor: '#fff',
            primaryBorderColor: '#ff6b6b',
            lineColor: '#333333',
            secondaryColor: '#ffcc02',
            tertiaryColor: '#fff'
        }
    });

    // Re-initialize mermaid after page load to catch any dynamically loaded content
    setTimeout(function() {
        mermaid.init();
    }, 100);
});