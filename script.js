document.getElementById('copy').addEventListener('click', async (event) => {
  await navigator.clipboard.writeText(document.getElementById('bibtex').innerText);
  event.currentTarget.textContent = 'Copied!';
  setTimeout(() => event.currentTarget.textContent = 'Copy citation', 1600);
});

// Keep each RGB/result pair aligned across autoplay, looping, and tab wake-up.
document.querySelectorAll('.sync-pair').forEach((pair) => {
  const [rgb, result] = pair.querySelectorAll('video');
  const sync = () => {
    if (Math.abs(rgb.currentTime - result.currentTime) > 0.12) result.currentTime = rgb.currentTime;
    if (!rgb.paused && result.paused) result.play().catch(() => {});
  };
  rgb.addEventListener('play', () => { result.currentTime = rgb.currentTime; result.play().catch(() => {}); });
  rgb.addEventListener('seeked', sync);
  rgb.addEventListener('timeupdate', sync);
});

const slider = document.getElementById('result-slider');
const slides = [...slider.children];
const dots = [...document.querySelectorAll('.slider-dots button')];
let activeSlide = 0;

const showSlide = (index) => {
  activeSlide = (index + slides.length) % slides.length;
  slider.scrollTo({ left: slides[activeSlide].offsetLeft - slider.offsetLeft, behavior: 'smooth' });
  dots.forEach((dot, i) => dot.classList.toggle('active', i === activeSlide));
};

document.getElementById('slider-prev').addEventListener('click', () => showSlide(activeSlide - 1));
document.getElementById('slider-next').addEventListener('click', () => showSlide(activeSlide + 1));
dots.forEach((dot, i) => dot.addEventListener('click', () => showSlide(i)));

let scrollTimer;
slider.addEventListener('scroll', () => {
  clearTimeout(scrollTimer);
  scrollTimer = setTimeout(() => {
    activeSlide = slides.reduce((best, slide, i) =>
      Math.abs(slide.offsetLeft - slider.scrollLeft) < Math.abs(slides[best].offsetLeft - slider.scrollLeft) ? i : best, 0);
    dots.forEach((dot, i) => dot.classList.toggle('active', i === activeSlide));
  }, 80);
});
