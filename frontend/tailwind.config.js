/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // OmniPdM brand: Dark Gray + Neon Green accent (#22ff88).
        // 직접 사용처는 className 의 임의값(arbitrary value) 으로 적기 때문에
        // 토큰 수준에서는 brand 가독성만 별칭으로 둠.
        brand: {
          accent: '#22ff88',
        },
      },
      gridTemplateColumns: {
        15: 'repeat(15, minmax(0, 1fr))',
      },
    },
  },
  plugins: [],
};
