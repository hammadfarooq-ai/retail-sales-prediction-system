/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
      colors: {
        brand: {
          50: '#eef4fd',
          100: '#cde2fb',
          500: '#2a78d6',
          600: '#256abf',
          700: '#1c5cab',
          900: '#0d366b',
        },
        // chart series (validated categorical order)
        s1: '#2a78d6',
        s2: '#eb6834',
        s3: '#1baf7a',
        s4: '#eda100',
      },
      boxShadow: {
        card: '0 1px 2px rgba(15,23,42,.05), 0 1px 3px rgba(15,23,42,.06)',
      },
    },
  },
  plugins: [],
}
