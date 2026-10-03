import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

const container = document.getElementById('root');
if (!container) throw new Error('#root not found in index.html');

// No StrictMode: its double-invoked effects would fire two real backend jobs
// (lens readouts, attribution) per user action in development.
createRoot(container).render(<App />);
