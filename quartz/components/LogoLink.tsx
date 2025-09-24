import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const LogoLink: QuartzComponent = ({ cfg }: QuartzComponentProps) => {
  return (
    <a href="https://ededdyedward.com" className="logo-link">
      <img src="/static/EdEddyEdward-Logo.png" alt="Eddy Zhou" />
    </a>
  )
}

LogoLink.css = `
.logo-link {
  display: inline-flex;
  background-color: #c03541;
  padding: 0.0rem 0.5rem;
  border-radius: 0.25rem;
  align-items: center;
  justify-content: center;
  transition: background-color;
  text-decoration: none;
  margin-bottom: 1rem;
}

.logo-link:hover {
  background-color: #f06571;
}

.logo-link img {
  height: 1.00rem;
  object-fit: contain;
  display: block;
}

/* Mobile styles - match hamburger menu height */
@media (max-width: 1200px) {
  .logo-link {
    padding: 5px;
    margin-bottom: 0;
    height: 34px;
    width: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
  }
  
  .logo-link img {
    height: 0.75rem;
    object-fit: contain;
  }
}
`

export default (() => LogoLink) satisfies QuartzComponentConstructor