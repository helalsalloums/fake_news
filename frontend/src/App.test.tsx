import {render, screen} from '@testing-library/react'
import {describe, expect, it} from 'vitest'
import App from './App'

describe('App', () => {
  it('renders the Arabic input experience', () => {
    render(<App />)
    expect(screen.getByRole('heading', {name: 'مدقّق الأخبار العربية'})).toBeInTheDocument()
    expect(screen.getByLabelText('ألصق الخبر أو الادعاء هنا').closest('main')).toHaveAttribute('dir', 'rtl')
    expect(screen.getByRole('button', {name: 'تحقق من الخبر'})).toBeDisabled()
  })
})
