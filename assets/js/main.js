// Main JavaScript for Pablo Rocamora's website

document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu functionality
    const mobileMenuButton = document.querySelector('.md\\:hidden');
    const mobileMenu = document.querySelector('.hidden.md\\:flex');
    
    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
            mobileMenu.classList.toggle('flex');
            mobileMenu.classList.toggle('flex-col');
            mobileMenu.classList.toggle('absolute');
            mobileMenu.classList.toggle('top-16');
            mobileMenu.classList.toggle('right-4');
            mobileMenu.classList.toggle('bg-gradient-to-r');
            mobileMenu.classList.toggle('from-blue-800');
            mobileMenu.classList.toggle('to-blue-900');
            mobileMenu.classList.toggle('p-4');
            mobileMenu.classList.toggle('rounded-lg');
            mobileMenu.classList.toggle('shadow-xl');
            mobileMenu.classList.toggle('space-y-4');
            mobileMenu.classList.toggle('z-10');
        });
    }

    // Form submission handling
    const contactForm = document.querySelector('#contact form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form values
            const name = document.querySelector('#name').value.trim();
            const email = document.querySelector('#email').value.trim();
            const message = document.querySelector('#message').value.trim();
            
            // Basic validation
            if (!name || !email || !message) {
                alert('Please fill in all fields');
                return;
            }
            
            if (!isValidEmail(email)) {
                alert('Please enter a valid email address');
                return;
            }
            
            // Here you would normally send the form data to a server
            // For now, we'll just show a success message
            alert('Thanks for your message! I will get back to you soon.');
            contactForm.reset();
        });
    }
    
    // Email validation helper
    function isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    // Skill bar animation on scroll
    const skillBars = document.querySelectorAll('.skill-bar');
    if (skillBars.length) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.2 });
        
        skillBars.forEach(bar => {
            observer.observe(bar);
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
                
                // Update URL without page reload
                history.pushState(null, null, targetId);
            }
        });
    });
    
    // Add active class to navigation based on scroll position
    window.addEventListener('scroll', function() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('nav a[href^="#"]');
        
        let currentSection = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (pageYOffset >= sectionTop - 200) {
                currentSection = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('text-blue-200');
            if (link.getAttribute('href') === `#${currentSection}`) {
                link.classList.add('text-blue-200');
            }
        });
    });
});