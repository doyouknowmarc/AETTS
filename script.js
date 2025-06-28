document.getElementById('github-btn').addEventListener('click', function() {
    window.open(this.dataset.url, '_blank');
});

const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.step').forEach(step => observer.observe(step));
